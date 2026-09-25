"""Backends: anything that can return the top log-probabilities of a model's first output token.

Every backend implements ``first_token(prompt, top_k) -> (tops, input_tokens)`` where ``tops`` is a list of
``(token_text, logprob)``. Only one token is ever generated; the prompt is a single prefill.
"""
from __future__ import annotations

import http.client
import json
import math
import re
import threading
import time
import urllib.parse
from typing import Dict, Iterator, List, Optional, Tuple

Tops = List[Tuple[str, float]]


class BackendError(RuntimeError):
    pass


class _HTTP:
    """``key_header``: where the key goes. "Authorization" (the default) sends ``Bearer <key>``; any other name sends
    the key as is (Azure OpenAI uses ``api-key``). ``headers``: extra headers for every request.

    Connections are kept alive, one per thread and host, so a decision doesn't pay a new TLS handshake per request.
    A connection the server has closed while idle is reopened once, transparently."""

    _STALE = (http.client.RemoteDisconnected, http.client.CannotSendRequest, ConnectionResetError, BrokenPipeError)

    def __init__(self, url: str, api_key: Optional[str] = None, timeout: float = 60.0, retries: int = 2,
                 key_header: str = "Authorization", headers: Optional[Dict[str, str]] = None):
        self.url, self.api_key, self.timeout, self.retries = url.rstrip("/"), api_key, timeout, retries
        self.key_header, self.headers = key_header, dict(headers or {})
        self._local = threading.local()

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            bearer = self.key_header.lower() == "authorization"
            headers[self.key_header] = f"Bearer {self.api_key}" if bearer else self.api_key
        return headers

    def _conn(self, scheme: str, netloc: str, fresh: bool = False):
        pool = self._local.__dict__.setdefault("pool", {})
        if fresh:
            self._drop(scheme, netloc)
        if (scheme, netloc) not in pool:
            cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
            pool[(scheme, netloc)] = cls(netloc, timeout=self.timeout)
        return pool[(scheme, netloc)]

    def _drop(self, scheme: str, netloc: str) -> None:
        c = self._local.__dict__.setdefault("pool", {}).pop((scheme, netloc), None)
        if c:
            c.close()

    def _send(self, path: str, body: dict):
        u = urllib.parse.urlsplit(self.url + path)
        target = (u.path or "/") + (f"?{u.query}" if u.query else "")
        data = json.dumps(body).encode()
        for fresh in (False, True):
            conn = self._conn(u.scheme, u.netloc, fresh)
            try:
                conn.request("POST", target, body=data, headers=self._headers())
                return conn.getresponse(), u
            except self._STALE:
                self._drop(u.scheme, u.netloc)
                if fresh:
                    raise
            except Exception:
                self._drop(u.scheme, u.netloc)
                raise

    def _response(self, path: str, body: dict):
        """A successful response (retrying overload and server errors, not client errors)."""
        for attempt in range(self.retries + 1):
            try:
                r, u = self._send(path, body)
            except Exception as e:
                if attempt == self.retries:
                    raise BackendError(f"{path}: {e}") from e
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status < 400:
                return r, u
            detail = r.read()[:300].decode("utf-8", "replace")
            if (r.status < 500 and r.status != 429) or attempt == self.retries:
                raise BackendError(f"{path} HTTP {r.status}: {detail}")
            time.sleep(1.5 * (attempt + 1))
        raise BackendError("unreachable")

    def post(self, path: str, body: dict) -> dict:
        r, u = self._response(path, body)
        try:
            return json.loads(r.read())
        except Exception as e:
            self._drop(u.scheme, u.netloc)
            raise BackendError(f"{path}: {e}") from e

    def stream(self, path: str, body: dict) -> Iterator[dict]:
        """Server-sent events as dicts. Stop early by closing the generator; the connection is then discarded,
        since a half-read stream can't be reused."""
        r, u = self._response(path, {**body, "stream": True})
        try:
            for line in r:
                line = line.strip()
                if line.startswith(b"data:"):
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        return
                    yield json.loads(data)
        finally:
            self._drop(u.scheme, u.netloc)


class OpenAIChat(_HTTP):
    """Any OpenAI-compatible chat completions endpoint that returns logprobs: OpenAI, vLLM, llama.cpp's llama-server,
    SGLang, and hosted providers that pass logprobs through.

    ``url`` may be the server root (``https://api.openai.com``), a base URL ending in ``/v1`` (what most SDKs call
    base_url), or the full ``.../chat/completions`` URL, query string included (Azure OpenAI).
    ``thinking="off"`` sends chat_template_kwargs {enable_thinking: false} (Qwen3-style templates on vLLM/llama-server).
    ``reasoning_effort="none"`` is what OpenAI's GPT-5.1-and-later models need before they return logprobs.

    Newer OpenAI models refuse some of the classic parameters. Each refusal is read from the error message, adapted to
    once, and remembered: ``max_tokens`` becomes ``max_completion_tokens`` (with room for their few hidden tokens),
    ``top_logprobs`` drops to the model's cap, ``temperature`` is left out, and reasoning is switched off."""

    def __init__(self, model: str, url: str = "https://api.openai.com", api_key: Optional[str] = None,
                 thinking: Optional[str] = None, reasoning_effort: Optional[str] = None,
                 extra_body: Optional[Dict] = None, **kw):
        """``extra_body``: more request fields, sent as they are (e.g. {"service_tier": "priority"})."""
        super().__init__(url, api_key, **kw)
        self.model, self.thinking, self.reasoning_effort = model, thinking, reasoning_effort
        self.extra_body = dict(extra_body or {})
        self.max_key, self.max_value = "max_tokens", 1
        self.top_cap: Optional[int] = None
        self.send_temperature = True
        self._lock = threading.Lock()

    def endpoint(self) -> str:
        if "/chat/completions" in self.url:
            return ""
        return "/chat/completions" if self.url.endswith("/v1") else "/v1/chat/completions"

    def _body(self, prompt: str, top_k: int) -> dict:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], self.max_key: self.max_value,
                "logprobs": True, "top_logprobs": min(int(top_k), 20, self.top_cap or 20)}
        if self.send_temperature:
            body["temperature"] = 0
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.thinking == "off":
            body["chat_template_kwargs"] = {"enable_thinking": False}
        body.update(self.extra_body)
        return body

    def _adapt(self, msg: str) -> bool:
        """Adjust to a refusal; False when there is nothing left to try."""
        if "max_completion_tokens" in msg and self.max_key == "max_tokens":
            self.max_key, self.max_value = "max_completion_tokens", 16
            return True
        cap = re.search(r"top_logprobs.{0,40}?less than or equal to (\d+)", msg)
        if cap and self.top_cap is None:
            self.top_cap = int(cap.group(1))
            return True
        if "temperature" in msg and self.send_temperature:
            self.send_temperature = False
            return True
        if ("output limit" in msg or "logprobs" in msg) and self.reasoning_effort is None:
            self.reasoning_effort = "none"
            return True
        if "output limit" in msg and self.max_value < 64:
            self.max_value *= 4
            return True
        return False

    def first_token(self, prompt: str, top_k: int = 20) -> Tuple[Tops, int]:
        for _ in range(6):
            body = self._body(prompt, top_k)
            try:
                r = self.post(self.endpoint(), body)
                break
            except BackendError as e:
                with self._lock:                    # parallel requests meet the same refusal: adapt once
                    if self._body(prompt, top_k) == body and not self._adapt(str(e)):
                        raise
        else:
            raise BackendError("the server kept refusing the request")
        try:
            first = r["choices"][0]["logprobs"]["content"][0]
        except (KeyError, IndexError, TypeError) as e:
            raise BackendError(f"no logprobs in the response (does this server support them?): {str(r)[:200]}") from e
        return [(t["token"], float(t["logprob"])) for t in first.get("top_logprobs", [])], \
            int((r.get("usage") or {}).get("prompt_tokens") or 0)


class LMStudio(_HTTP):
    """LM Studio: its chat endpoint returns no logprobs, but /v1/responses does (with reasoning switched off)."""

    def __init__(self, model: str, url: str = "http://127.0.0.1:1234", **kw):
        super().__init__(url, None, **kw)
        self.model = model

    def first_token(self, prompt: str, top_k: int = 20) -> Tuple[Tops, int]:
        r = self.post("/v1/responses", {"model": self.model, "input": prompt, "max_output_tokens": 2,
                                        "temperature": 0, "top_logprobs": min(int(top_k), 20),
                                        "reasoning": {"effort": "none"},
                                        "include": ["message.output_text.logprobs"]})
        msgs = [o for o in r.get("output", []) if o.get("type") == "message"]
        if not msgs or not msgs[0]["content"][0].get("logprobs"):
            raise BackendError(f"no logprobs in the response: {str(r)[:200]}")
        first = msgs[0]["content"][0]["logprobs"][0]
        return [(t["token"], float(t["logprob"])) for t in first.get("top_logprobs", [])], \
            int((r.get("usage") or {}).get("input_tokens") or 0)


class Anthropic(_HTTP):
    """Anthropic's Messages API. It returns no log-probabilities, so this backend reports the answer the model gives:
    one reply per option order, with the answer letter taken from the reply. Recent Claude models reason before they
    answer (adaptive thinking, which the newest can't switch off), so the reply may include reasoning: thinking is
    switched off where the model allows it, and otherwise set to its lowest effort. This is a different mode from the
    one-token logprob readout, slower and not calibrated: the probabilities are just the answer (``readout`` says so).
    ``samples`` > 1 asks several times at the default temperature and uses the answer frequencies."""

    readout = "answer"
    _LINE = re.compile(r"^\W*([A-Z])\W*$")
    _SAID = re.compile(r"(?i:answer|option|choice)\W{0,4}(?:(?i:is)\W{0,4})?\(?([A-Z])\b(?!\w)")

    def __init__(self, model: str, url: str = "https://api.anthropic.com", api_key: Optional[str] = None,
                 samples: int = 1, max_tokens: int = 2048, version: str = "2023-06-01", stream: bool = True, **kw):
        """``stream``: read the reply as it is written and stop as soon as it opens with a letter on its own line,
        instead of waiting for any explanation after it."""
        kw.setdefault("key_header", "x-api-key")
        kw.setdefault("headers", {"anthropic-version": version})
        super().__init__(url, api_key, **kw)
        self.model, self.samples, self.max_tokens, self.stream_replies = model, max(1, int(samples)), max_tokens, stream
        self.send_temperature = True               # the newest models refuse it
        self.thinking: Optional[dict] = {"thinking": {"type": "disabled"}}
        self._lock = threading.Lock()

    def _body(self, prompt: str) -> dict:
        body = {"model": self.model, "max_tokens": self.max_tokens, "messages": [{"role": "user", "content": prompt}],
                **(self.thinking or {})}
        if self.send_temperature and self.samples == 1:
            body["temperature"] = 0
        return body

    def _adapt(self, msg: str) -> bool:
        if "temperature" in msg and self.send_temperature:
            self.send_temperature = False
            return True
        if "thinking" in msg and self.thinking and self.thinking["thinking"]["type"] == "disabled":
            self.thinking = ({"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}
                             if "adaptive" in msg else None)
            return True
        if "effort" in msg and self.thinking and "output_config" in self.thinking:
            self.thinking = {"thinking": {"type": "adaptive"}}
            return True
        if "thinking" in msg and self.thinking:
            self.thinking = None
            return True
        return False

    def _call(self, prompt: str) -> Tuple[str, int]:
        for _ in range(5):
            body = self._body(prompt)
            try:
                return self._reply(body) if self.stream_replies else self._whole(body)
            except BackendError as e:
                with self._lock:
                    if self._body(prompt) == body and not self._adapt(str(e)):
                        raise
        raise BackendError("the server kept refusing the request")

    def _whole(self, body: dict) -> Tuple[str, int]:
        r = self.post("/v1/messages", body)
        text = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
        return text, int((r.get("usage") or {}).get("input_tokens") or 0)

    def _reply(self, body: dict) -> Tuple[str, int]:
        text, tokens = "", 0
        events = self.stream("/v1/messages", body)
        try:
            for ev in events:
                kind = ev.get("type")
                if kind == "message_start":
                    tokens = int(((ev.get("message") or {}).get("usage") or {}).get("input_tokens") or 0)
                elif kind == "content_block_delta" and (ev.get("delta") or {}).get("type") == "text_delta":
                    text += ev["delta"].get("text", "")
                    head = text.lstrip()
                    if "\n" in head and self._LINE.match(head.split("\n", 1)[0]):
                        break                               # the answer is in: skip the explanation
                elif kind == "error":
                    raise BackendError(f"stream error: {ev.get('error')}")
        finally:
            events.close()
        return text, tokens

    @classmethod
    def letter(cls, text: str) -> Optional[str]:
        """The answer letter in a reply: a reply that is just the letter, else an 'answer is X', else the last line
        that is only a letter."""
        text = text.strip()
        if cls._LINE.match(text.split("\n", 1)[0]):
            return cls._LINE.match(text.split("\n", 1)[0]).group(1)
        said = cls._SAID.findall(text)
        if said:
            return said[-1]
        for line in reversed(text.splitlines()):
            m = cls._LINE.match(line)
            if m:
                return m.group(1)
        return None

    def first_token(self, prompt: str, top_k: int = 20) -> Tuple[Tops, int]:
        counts: Dict[str, int] = {}
        tokens = 0
        for _ in range(self.samples):
            text, tokens = self._call(prompt)
            got = self.letter(text)
            if got:
                counts[got] = counts.get(got, 0) + 1
        if not counts:
            raise BackendError(f"no answer letter in the reply: {text[:120]!r}")
        return [(t, math.log(c / self.samples)) for t, c in sorted(counts.items(), key=lambda kv: -kv[1])], tokens


class LlamaCpp(_HTTP):
    """llama.cpp's native /completion endpoint (n_probs). The prompt is sent raw, so wrap it in the model's chat
    template yourself (``template``: a format string with {prompt}), or use OpenAIChat against llama-server."""

    def __init__(self, url: str = "http://127.0.0.1:8080", template: str = "{prompt}", **kw):
        super().__init__(url, None, **kw)
        self.template = template

    def first_token(self, prompt: str, top_k: int = 20) -> Tuple[Tops, int]:
        r = self.post("/completion", {"prompt": self.template.format(prompt=prompt), "n_predict": 1,
                                      "n_probs": int(top_k), "temperature": 0, "post_sampling_probs": False})
        try:
            probs = r["completion_probabilities"][0]
        except (KeyError, IndexError) as e:
            raise BackendError(f"no completion_probabilities: {str(r)[:200]}") from e
        tops = probs.get("top_logprobs") or probs.get("probs") or []
        out = []
        for t in tops:
            tok = t.get("token", t.get("tok_str", ""))
            lp = t.get("logprob")
            if lp is None and t.get("prob") is not None:
                import math
                lp = math.log(max(float(t["prob"]), 1e-12))
            out.append((tok, float(lp)))
        return out, int(r.get("tokens_evaluated") or 0)


class Transformers:
    """A local Hugging Face model: the logits at the last prompt position, in-process (pip install jevless[local])."""

    def __init__(self, model: str, device: Optional[str] = None, dtype: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype,
                                                          device_map=device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()

    def first_token(self, prompt: str, top_k: int = 20) -> Tuple[Tops, int]:
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            logits = self.model(**ids).logits[0, -1].float()
        lp = self.torch.log_softmax(logits, -1)
        vals, idx = lp.topk(int(top_k))
        return [(self.tok.decode([int(i)]), float(v)) for v, i in zip(vals, idx)], int(ids["input_ids"].shape[1])
