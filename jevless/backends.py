"""Backends: anything that can return the top log-probabilities of a model's first output token.

Every backend implements ``first_token(prompt, top_k) -> (tops, input_tokens)`` where ``tops`` is a list of
``(token_text, logprob)``. Only one token is ever generated; the prompt is a single prefill.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

Tops = List[Tuple[str, float]]


class BackendError(RuntimeError):
    pass


class _HTTP:
    def __init__(self, url: str, api_key: Optional[str] = None, timeout: float = 60.0, retries: int = 2):
        self.url, self.api_key, self.timeout, self.retries = url.rstrip("/"), api_key, timeout, retries

    def post(self, path: str, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:          # retry overload and server errors, not client errors
                detail = e.read()[:300].decode("utf-8", "replace")
                if (e.code < 500 and e.code != 429) or attempt == self.retries:
                    raise BackendError(f"{path} HTTP {e.code}: {detail}") from e
            except Exception as e:
                if attempt == self.retries:
                    raise BackendError(f"{path}: {e}") from e
            time.sleep(1.5 * (attempt + 1))
        raise BackendError("unreachable")


class OpenAIChat(_HTTP):
    """Any OpenAI-compatible /v1/chat/completions that returns logprobs: OpenAI (non-reasoning models), vLLM,
    llama.cpp's llama-server, SGLang, and hosted providers that pass logprobs through.
    ``thinking="off"`` sends chat_template_kwargs {enable_thinking: false} (Qwen3-style templates on vLLM/llama-server)."""

    def __init__(self, model: str, url: str = "https://api.openai.com", api_key: Optional[str] = None,
                 thinking: Optional[str] = None, **kw):
        super().__init__(url, api_key, **kw)
        self.model, self.thinking = model, thinking

    def first_token(self, prompt: str, top_k: int = 20) -> Tuple[Tops, int]:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1,
                "temperature": 0, "logprobs": True, "top_logprobs": min(int(top_k), 20)}
        if self.thinking == "off":
            body["chat_template_kwargs"] = {"enable_thinking": False}
        r = self.post("/v1/chat/completions", body)
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
