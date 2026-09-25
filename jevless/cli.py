"""jevless command line.

  jevless probe --backend lmstudio --model qwen/qwen3.5-9b          # does this model/server give usable readouts?
  jevless ask --backend openai --model gpt-4.1-mini --state "The API returns 503 since the deploy." \\
      --choice "Which team should own this?" billing infra sales
  jevless serve --backend lmstudio --model qwen/qwen3.5-9b --port 8765
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import Decider, LlamaCpp, LMStudio, OpenAIChat


def backend_from(args):
    if args.backend == "openai":
        return OpenAIChat(args.model, url=args.url or "https://api.openai.com",
                          api_key=args.api_key or os.environ.get("OPENAI_API_KEY"), thinking=args.thinking)
    if args.backend == "lmstudio":
        return LMStudio(args.model, url=args.url or "http://127.0.0.1:1234")
    if args.backend == "llamacpp":
        return LlamaCpp(url=args.url or "http://127.0.0.1:8080")
    if args.backend == "transformers":
        from . import Transformers
        return Transformers(args.model)
    raise SystemExit(f"unknown backend {args.backend}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jevless", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["probe", "ask", "serve"])
    ap.add_argument("--backend", default="openai", choices=["openai", "lmstudio", "llamacpp", "transformers"])
    ap.add_argument("--model", default="")
    ap.add_argument("--url", default="")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--thinking", default=None, choices=[None, "off"], help="off: disable Qwen3-style thinking")
    ap.add_argument("--orders", type=int, default=2, help="1 or 2 option orders (2 cancels position bias)")
    ap.add_argument("--state", default="")
    ap.add_argument("--noul", default="")
    ap.add_argument("--choice", nargs="+", default=None, metavar=("QUESTION", "OPTION"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    decider = Decider(backend_from(args), permutations=args.orders)

    if args.command == "probe":
        checks = [("noul, clearly true", "The sky is blue on a clear day.", "The statement is about weather or the sky.", True),
                  ("noul, clearly false", "The sky is blue on a clear day.", "The statement is about cooking.", False)]
        ok = True
        for name, state, q, want in checks:
            a = decider.noul(state, q)
            good = (a.noul > 0.5) == want
            ok &= good
            print(f"{'ok ' if good else 'BAD'} {name}: noul={a.noul:.3f} flip={a.flip} {a.latency_ms:.0f} ms")
        a = decider.choice("A customer writes: my card was charged twice.", "Which team should handle this?",
                           {"billing": None, "shipping": None, "security": None})
        ok &= a.choice == "billing"
        print(f"{'ok ' if a.choice == 'billing' else 'BAD'} choice: {a.choice} "
              f"{ {k: round(v, 3) for k, v in a.probabilities.items()} } {a.latency_ms:.0f} ms")
        sys.exit(0 if ok else 1)
    if args.command == "ask":
        if args.noul:
            a = decider.noul(args.state, args.noul)
        elif args.choice and len(args.choice) >= 3:
            a = decider.choice(args.state, args.choice[0], {o: None for o in args.choice[1:]})
        else:
            raise SystemExit("give --noul STATEMENT or --choice QUESTION OPTION OPTION [...]")
        print(json.dumps(a.to_wire(), indent=1))
        return
    from .server import make_server
    srv = make_server(decider, args.host, args.port, api_key=os.environ.get("JEVLESS_API_KEY"))
    print(f"jevless: POST http://{args.host}:{args.port}/v1/systemone ({args.backend} {args.model})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
