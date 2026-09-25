"""jevless command line.

  jevless probe --backend lmstudio --model qwen/qwen3.5-9b          # does this model/server give usable readouts?
  jevless bench --backend openai --model gpt-4.1-mini              # accuracy and calibration on 30 labelled decisions
  jevless ask --backend openai --model gpt-4.1-mini --state "The API returns 503 since the deploy." \\
      --choice "Which team should own this?" billing infra sales
  jevless serve --backend lmstudio --model qwen/qwen3.5-9b --port 8765

Any OpenAI-compatible server, with the key read from the environment variable you name:

  jevless bench --backend openai --url https://llm.example.com/v1 --model my-model --api-key-env MY_LLM_KEY

--url defaults to $OPENAI_BASE_URL, then https://api.openai.com. The key comes from --api-key-env (default
OPENAI_API_KEY); --api-key puts it on the command line instead, where other users on the machine can see it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import BackendError, Decider, DecisionError, LlamaCpp, LMStudio, OpenAIChat


def _headers(pairs):
    out = {}
    for p in pairs or []:
        name, sep, value = p.partition(":")
        if not sep:
            raise SystemExit(f"--header wants 'Name: value', got {p!r}")
        out[name.strip()] = value.strip()
    return out


def backend_from(args):
    if args.backend == "openai":
        key = args.api_key or os.environ.get(args.api_key_env)
        if args.api_key_env != "OPENAI_API_KEY" and not key:
            raise SystemExit(f"jevless: ${args.api_key_env} is not set")
        return OpenAIChat(args.model, url=args.url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com",
                          api_key=key, thinking=args.thinking, key_header=args.key_header,
                          headers=_headers(args.header))
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
    ap.add_argument("command", choices=["probe", "bench", "ask", "serve"])
    ap.add_argument("--backend", default="openai", choices=["openai", "lmstudio", "llamacpp", "transformers"])
    ap.add_argument("--model", default="")
    ap.add_argument("--url", default="")
    ap.add_argument("--api-key", default="", help="the key itself (visible in the process list; prefer --api-key-env)")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", metavar="NAME", help="environment variable holding the key")
    ap.add_argument("--key-header", default="Authorization",
                    help="header for the key: Authorization sends 'Bearer <key>', any other name the bare key (Azure: api-key)")
    ap.add_argument("--header", action="append", metavar="'NAME: VALUE'", help="extra request header (repeatable)")
    ap.add_argument("--file", default="", help="bench: your own labelled decisions, JSONL (see jevless.bench)")
    ap.add_argument("--out", default="", help="bench: write every decision and the summary as JSON")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests")
    ap.add_argument("--thinking", default=None, choices=[None, "off"], help="off: disable Qwen3-style thinking")
    ap.add_argument("--orders", type=int, default=2, help="1 or 2 option orders (2 cancels position bias)")
    ap.add_argument("--state", default="")
    ap.add_argument("--noul", default="")
    ap.add_argument("--choice", nargs="+", default=None, metavar=("QUESTION", "OPTION"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    decider = Decider(backend_from(args), permutations=args.orders, workers=args.workers)
    try:
        run(args, decider)
    except (BackendError, DecisionError) as e:
        hint = ""
        if "logprobs" in str(e) or "top_logprobs" in str(e):
            hint = ("\n(reasoning models, such as OpenAI's o-series, return no logprobs: use a chat model like "
                    "gpt-4.1-mini, or switch thinking off)")
        raise SystemExit(f"jevless: {e}{hint}")


def run(args, decider):
    if args.command == "bench":
        from . import bench
        items = bench.load(args.file) if args.file else bench.BUILTIN
        result = bench.run(decider, items, workers=args.workers)
        print(f"{args.backend} {args.model or ''} {args.url or ''}".strip())
        print(bench.report(result))
        if args.out:
            with open(args.out, "w") as fh:
                json.dump({"backend": args.backend, "model": args.model, "url": args.url, "orders": args.orders,
                           **result}, fh, indent=1)
        return

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
