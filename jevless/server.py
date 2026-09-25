"""A local /v1/systemone endpoint: Jev-style decision requests answered by your own model.

  jevless serve --backend lmstudio --model qwen/qwen3.5-9b --port 8765
  curl -s localhost:8765/v1/systemone -d '{"state": "...", "questions": {"q1": {"type": "noul", "instructions": "..."}}}'

Errors follow the same conventions: 422 for a malformed request, 529 when the model backend fails.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from .backends import BackendError
from .core import DecisionError, Decider


def make_server(decider: Decider, host: str = "127.0.0.1", port: int = 8765, api_key: Optional[str] = None):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") in ("/health", "/v1/health"):
                return self._send(200, {"ok": True})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self._send(404, {"error": "POST /v1/systemone"})
            if api_key and self.headers.get("Authorization") != f"Bearer {api_key}":
                return self._send(401, {"error": "invalid or missing API key"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                self._send(200, decider.system_one(req))
            except (ValueError, DecisionError, KeyError, TypeError) as e:
                if isinstance(e, DecisionError) and "top tokens" in str(e):
                    return self._send(529, {"error": str(e)})
                self._send(422, {"error": str(e)})
            except BackendError as e:
                self._send(529, {"error": f"model backend failed: {e}"})

        def log_message(self, fmt, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)
