import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from jevless import Decider, OpenAIChat, bench
from jevless.cli import main


class Stub:
    """An OpenAI-compatible chat endpoint that answers 'A' and records what it was sent."""
    def __init__(self, want_completion_tokens=False):
        self.seen = []
        stub = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.seen.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                                  "body": body})
                if want_completion_tokens and "max_tokens" in body:
                    data = json.dumps({"error": {"message": "Unsupported parameter: 'max_tokens'. "
                                                            "Use 'max_completion_tokens' instead."}}).encode()
                    self.send_response(400)
                else:
                    data = json.dumps({"choices": [{"logprobs": {"content": [{"token": "A", "logprob": -0.1,
                                      "top_logprobs": [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.5}]}]}}],
                                       "usage": {"prompt_tokens": 42}}).encode()
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
        assert be.max_key == "max_completion_tokens" and s.seen[-1]["body"]["max_completion_tokens"] == 1
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
    assert len(bench.BUILTIN) == 30 and all(bench._gold(i) for i in bench.BUILTIN)
