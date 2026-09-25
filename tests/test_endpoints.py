import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from jevless import Decider, OpenAIChat, bench
from jevless.cli import main


class Stub:
    """An OpenAI-compatible chat endpoint that answers 'A' and records what it was sent."""
    def __init__(self, want_completion_tokens=False, gpt5=False):
        self.seen = []
        stub = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.seen.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                                  "body": body})
                refusal = None
                if (want_completion_tokens or gpt5) and "max_tokens" in body:
                    refusal = "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead."
                elif gpt5 and body.get("top_logprobs", 0) > 5:
                    refusal = "Invalid value for 'top_logprobs': must be less than or equal to 5."
                elif gpt5 and "temperature" in body:
                    refusal = "Unsupported value: 'temperature' does not support 0 with this model."
                elif gpt5 and body.get("reasoning_effort") != "none":
                    refusal = "Could not finish the message because max_tokens or model output limit was reached."
                if refusal:
                    data = json.dumps({"error": {"message": refusal}}).encode()
                    self.send_response(400)
                else:
                    data = json.dumps({"choices": [{"logprobs": {"content": [{"token": "A", "logprob": -0.1,
                                      "top_logprobs": [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.5}]}]}}],
                                       "usage": {"prompt_tokens": 42, "completion_tokens": 1}}).encode()
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.root = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def close(self):
        self.srv.shutdown()


@pytest.fixture
def stub():
    s = Stub()
    yield s
    s.close()


@pytest.mark.parametrize("suffix,path", [("", "/v1/chat/completions"), ("/v1", "/v1/chat/completions"),
                                         ("/openai/deployments/d1/chat/completions?api-version=2024-10-21",
                                          "/openai/deployments/d1/chat/completions?api-version=2024-10-21")])
def test_url_forms(stub, suffix, path):
    Decider(OpenAIChat("m", url=stub.root + suffix), permutations=1).noul("s", "q")
    assert stub.seen[-1]["path"] == path


def test_key_headers(stub):
    Decider(OpenAIChat("m", url=stub.root, api_key="k1"), permutations=1).noul("s", "q")
    assert stub.seen[-1]["headers"]["authorization"] == "Bearer k1"
    Decider(OpenAIChat("m", url=stub.root, api_key="k2", key_header="api-key", headers={"X-Org": "o"}),
            permutations=1).noul("s", "q")
    h = stub.seen[-1]["headers"]
    assert h["api-key"] == "k2" and "authorization" not in h and h["x-org"] == "o"


def test_max_completion_tokens_fallback():
    s = Stub(want_completion_tokens=True)
    try:
        be = OpenAIChat("m", url=s.root)
        Decider(be, permutations=1).noul("s", "q")
        assert be.max_key == "max_completion_tokens" and s.seen[-1]["body"]["max_completion_tokens"] == 16
    finally:
        s.close()


def test_cli_reads_the_named_env_var(stub, monkeypatch, capsys):
    monkeypatch.setenv("MY_LLM_KEY", "secret-from-env")
    main(["ask", "--url", stub.root + "/v1", "--model", "m", "--api-key-env", "MY_LLM_KEY", "--orders", "1",
          "--state", "s", "--noul", "q"])
    assert stub.seen[-1]["headers"]["authorization"] == "Bearer secret-from-env"
    assert "noul" in capsys.readouterr().out
    monkeypatch.delenv("MY_LLM_KEY")
    with pytest.raises(SystemExit, match="MY_LLM_KEY is not set"):
        main(["ask", "--url", stub.root, "--model", "m", "--api-key-env", "MY_LLM_KEY", "--state", "s", "--noul", "q"])


def test_bench_scores_and_reports():
    class Always:                                   # answers the first option, always
        def first_token(self, prompt, top_k=20):
            return [("A", -0.05), ("B", -3.0), ("C", -4.0), ("D", -5.0), ("E", -6.0)], 50
    items = [{"type": "noul", "state": "s", "instructions": "q", "gold": False},       # shown as A) false: right
             {"type": "choice", "state": "s", "instructions": "q", "criteria": {"x": None, "y": None}, "gold": "y"}]
    r = bench.run(Decider(Always(), permutations=1), items)
    assert r["overall"]["n"] == 2 and r["overall"]["accuracy"] == 0.5
    assert "miss [choice]" in bench.report(r)
    for items in (bench.BUILTIN, bench.HARD):
        assert len(items) == 30 and all(bench._gold(i) for i in items)
        for it in items:                                # every gold answer is one of its options
            from jevless.core import _options, question_from_wire
            assert bench._gold(it) in _options(question_from_wire(it))[0]


def test_adapts_to_gpt5_style_refusals():
    s = Stub(gpt5=True)
    try:
        be = OpenAIChat("m", url=s.root)
        a = Decider(be, permutations=1).noul("s", "q")
        body = s.seen[-1]["body"]
        assert a.probabilities and body["max_completion_tokens"] == 16 and body["top_logprobs"] == 5
        assert "temperature" not in body and body["reasoning_effort"] == "none"
        n = len(s.seen)
        Decider(be, permutations=1).noul("s", "q")               # remembered: one request, no refusals
        assert len(s.seen) == n + 1
    finally:
        s.close()


def test_parallel_requests_adapt_once():
    s = Stub(gpt5=True)
    try:
        items = [{"type": "noul", "state": f"s{i}", "instructions": "q", "gold": True} for i in range(12)]
        r = bench.run(Decider(OpenAIChat("m", url=s.root)), items, workers=8)
        assert r["overall"]["errors"] == 0
    finally:
        s.close()


def test_anthropic_answers_only():
    from jevless import Anthropic
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(({k.lower(): v for k, v in self.headers.items()}, body, self.path))
            first = re.search(r"^([A-Z])\) true", body["messages"][0]["content"], re.M).group(1)   # always says "true"
            if not body.get("stream"):
                data = json.dumps({"content": [{"type": "text", "text": first}], "usage": {"input_tokens": 30}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_response(200)                         # the letter, then a slow explanation
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            events = [{"type": "message_start", "message": {"usage": {"input_tokens": 30}}},
                      {"type": "content_block_delta", "delta": {"type": "text_delta", "text": first + "\n\n"}}]
            events += [{"type": "content_block_delta", "delta": {"type": "text_delta", "text": "because... "}}] * 5
            try:
                for i, ev in enumerate(events):
                    if i > 1:
                        time.sleep(0.2)
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        be = Anthropic("claude-x", url=f"http://127.0.0.1:{srv.server_address[1]}", api_key="ak")
        t = time.perf_counter()
        a = Decider(be).noul("s", "q")
        assert time.perf_counter() - t < 0.6                 # stopped at the letter, not after the explanation
        h, body, path = seen[-1]
        assert path == "/v1/messages" and h["x-api-key"] == "ak" and h["anthropic-version"] and "logprobs" not in body
        assert a.noul > 0.9 and not a.flip
        r = bench.run(Decider(be), [{"type": "noul", "state": "s", "instructions": "q", "gold": True}])
        assert r["overall"]["accuracy"] == 1.0 and r["overall"]["log_loss"] is None and r["readout"] == "answer"
        assert "answer mode" in bench.report(r)
        whole = Anthropic("claude-x", url=f"http://127.0.0.1:{srv.server_address[1]}", api_key="ak", stream=False)
        assert Decider(whole).noul("s", "q").noul > 0.9
    finally:
        srv.shutdown()


@pytest.mark.parametrize("reply,letter", [("B", "B"), ("  **C**\n\nbecause", "C"), ("36 + 16 = 52.\n\nAnswer: B", "B"),
                                          ("The answer is (A).", "A"), ("Let me think.\nIt holds.\nB", "B"),
                                          ("I need to calculate the total", None),
                                          ("The answer is a bit subtle.\n\nB", "B")])
def test_anthropic_letter_parsing(reply, letter):
    from jevless import Anthropic
    assert Anthropic.letter(reply) == letter


def test_connections_are_kept_alive():
    ports = []

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"                    # persistent connections

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            ports.append(self.client_address[1])
            data = json.dumps({"choices": [{"logprobs": {"content": [{"token": "A", "logprob": -0.1,
                              "top_logprobs": [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.0}]}]}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        d = Decider(OpenAIChat("m", url=f"http://127.0.0.1:{srv.server_address[1]}"), permutations=1)
        for _ in range(4):
            d.noul("s", "q")
        assert len(ports) == 4 and len(set(ports)) == 1
        ports.clear()                                    # both orders in parallel: still a few long-lived connections
        d2 = Decider(OpenAIChat("m", url=f"http://127.0.0.1:{srv.server_address[1]}"), workers=1)
        for _ in range(6):
            d2.noul("s", "q")
        assert len(ports) == 12 and len(set(ports)) <= 2
    finally:
        srv.shutdown()


def test_extra_body_is_sent(stub):
    Decider(OpenAIChat("m", url=stub.root, extra_body={"service_tier": "priority"}), permutations=1).noul("s", "q")
    assert stub.seen[-1]["body"]["service_tier"] == "priority"


def test_tokens_are_counted_per_decision(stub):
    a = Decider(OpenAIChat("m", url=stub.root)).noul("s", "q")          # two orders
    assert a.input_tokens == 84 and a.output_tokens == 2
