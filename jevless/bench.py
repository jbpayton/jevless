"""A quick labelled comparison of models: accuracy and calibration on small agent decisions.

  jevless bench --backend openai --model gpt-4.1-mini
  jevless bench --backend openai --model gpt-4.1-mini --file my_decisions.jsonl --out results.json

The built-in set is 30 hand-written decisions (noul, choice, score) with traps: negation, sarcasm, a near-miss
memory, a reply with nothing behind it, pronouns that need the rest of the sentence. It is a smoke test for comparing
models on the same footing, not a benchmark to publish. Bring your own decisions as JSONL in the /v1/systemone
question shape plus the right answer:

  {"state": "...", "type": "noul", "instructions": "...", "gold": true}
  {"state": "...", "type": "choice", "instructions": "...", "criteria": {"billing": null, "shipping": null}, "gold": "billing"}
  {"state": "...", "type": "score", "instructions": "...", "criteria": ["low", "mid", "high"], "gold": 2}
"""
from __future__ import annotations

import json
import math
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Mapping, Sequence

from .backends import BackendError
from .core import Decider, DecisionError, question_from_wire

_TEAMS = {"billing": None, "shipping": None, "security": None, "sales": None}
_URGENCY = ["can wait", "this week", "today", "right now"]
_MOOD = ["calm", "annoyed", "angry"]
_ANSWERS = "The memory answers the question."


def _n(state, statement, gold):
    return {"type": "noul", "state": state, "instructions": statement, "gold": gold}


def _c(state, question, options, gold):
    return {"type": "choice", "state": state, "instructions": question,
            "criteria": options if isinstance(options, dict) else {o: None for o in options}, "gold": gold}


def _s(state, question, levels, gold):
    return {"type": "score", "state": state, "instructions": question, "criteria": levels, "gold": gold}


BUILTIN: List[Dict[str, Any]] = [
    _n("The deploy at 14:02 went fine. The 503s started at 15:30, after the cache node rebooted.",
       "The deploy caused the 503 errors.", False),
    _n("Heads up: our sync moved from Tuesday to Thursday this week.", "The sync is on Tuesday this week.", False),
    _n("I finally got my refund, but it took three phone calls.", "The customer received a refund.", True),
    _n("Great, another outage. Exactly what I needed on launch day.", "The customer is happy.", False),
    _n("Memory: \"Sam's birthday is March 3.\"\nQuestion: \"When is Sam's birthday?\"", _ANSWERS, True),
    _n("Memory: \"Sam's sister Ana has her birthday on March 3.\"\nQuestion: \"When is Sam's birthday?\"",
       _ANSWERS, False),
    _n("Invoice total: $1,240. Approved budget: $1,200.", "The invoice is within budget.", False),
    _n("Ticket: \"Hi, what are your office hours on Saturdays?\"", "This ticket reports a bug.", False),
    _n("Log: ERROR disk /dev/sda1 is 98% full", "The machine is running low on disk space.", True),
    _n("User: \"I'm vegetarian, but I do eat fish.\"", "The user eats fish.", True),
    _n("Tool results this turn: none. Memory given: none.\nAssistant's reply: \"Your flight leaves at 9:40 from gate B12.\"",
       "Everything specific in the reply is backed by the tool results or the memory.", False),
    _n("Email: \"Please don't cancel my subscription, I only wanted to update the card on file.\"",
       "The customer wants to cancel the subscription.", False),
    _n("Alice is taller than Bob. Bob is taller than Carol.", "Carol is taller than Alice.", False),
    _n("Tracking: delivered 10:14, left with neighbour, signed by J. Ruiz.", "The package was delivered.", True),
    _c("A customer writes: my card was charged twice for one order.", "Which team should handle this?", _TEAMS, "billing"),
    _c("A customer writes: someone logged into my account from another country and changed my password.",
       "Which team should handle this?", _TEAMS, "security"),
    _c("A customer writes: the box arrived crushed and the mug inside is broken.",
       "Which team should handle this?", _TEAMS, "shipping"),
    _c("A customer writes: do you offer volume discounts for 200 seats?", "Which team should handle this?", _TEAMS,
       "sales"),
    _c("\"Wo ist der Bahnhof?\"", "Which language is this?", ["English", "French", "Spanish", "Italian", "German"],
       "German"),
    _c("Every API request has returned 401 since we rotated the key yesterday.", "What is the most likely cause?",
       ["the provider is down", "the client still sends the old key", "we hit a rate limit", "DNS is failing"],
       "the client still sends the old key"),
    _c("The trophy didn't fit in the suitcase because it was too big.", "What does \"it\" refer to?",
       ["the trophy", "the suitcase"], "the trophy"),
    _c("The trophy didn't fit in the suitcase because it was too small.", "What does \"it\" refer to?",
       ["the trophy", "the suitcase"], "the suitcase"),
    _c("Today is Friday. Memories:\n- Tuesday: \"Parked on level 3, row F.\"\n- Friday 9:05: \"Parked on level 2, row C.\"\n"
       "- Monday: \"The garage opens at 7am.\"\nThe user asks: \"Where did I park?\"", "Which memory answers it?",
       ["the Tuesday note", "the Friday note", "the Monday note"], "the Friday note"),
    _c("CI run: 1) git pull: ok 2) npm ci: ok 3) npm test: 2 failing 4) deploy: skipped", "Which step failed?",
       ["pull", "install", "test", "deploy"], "test"),
    _s("Our checkout page has been down for all customers for the last hour.", "How urgent is this?", _URGENCY, 3),
    _s("There's a typo on the About page: \"recieve\". No rush at all.", "How urgent is this?", _URGENCY, 0),
    _s("It's Wednesday. Please update the pricing page before Monday's launch.", "How urgent is this?", _URGENCY, 1),
    _s("\"Thanks so much, that fixed it!\"", "How upset is the customer?", _MOOD, 0),
    _s("\"Still waiting on that update. This is getting frustrating.\"", "How upset is the customer?", _MOOD, 1),
    _s("\"This is the THIRD time I'm writing. Unacceptable. Refund me NOW.\"", "How upset is the customer?", _MOOD, 2),
]


def load(path: str) -> List[Dict[str, Any]]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _gold(item: Mapping[str, Any]) -> str:
    g = item["gold"]
    return ("true" if g else "false") if item["type"] == "noul" else str(g)


def _one(decider: Decider, item: Mapping[str, Any]) -> Dict[str, Any]:
    gold = _gold(item)
    try:
        a = decider.ask(item["state"], question_from_wire(item))
    except (DecisionError, BackendError) as e:
        return {"type": item["type"], "instructions": item["instructions"], "gold": gold, "error": str(e)[:300]}
    probs = a.probabilities
    if gold not in probs:
        raise DecisionError(f"gold {gold!r} is not one of the options {list(probs)}")
    return {"type": item["type"], "instructions": item["instructions"], "gold": gold, "answer": a.choice,
            "correct": a.choice == gold, "p_gold": probs[gold], "confidence": a.confidence, "flip": a.flip,
            "brier": sum((p - (k == gold)) ** 2 for k, p in probs.items()),
            "latency_ms": a.latency_ms, "input_tokens": a.input_tokens}


def _stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return {"n": len(rows), "errors": len(rows)}
    return {"n": len(rows), "errors": len(rows) - len(ok),
            "accuracy": round(sum(r["correct"] for r in ok) / len(ok), 3),
            "log_loss": round(sum(-math.log(max(r["p_gold"], 1e-9)) for r in ok) / len(ok), 3),
            "brier": round(sum(r["brier"] for r in ok) / len(ok), 3),
            "flip_rate": round(sum(r["flip"] for r in ok) / len(ok), 3),
            "p50_ms": round(statistics.median(r["latency_ms"] for r in ok)),
            "mean_input_tokens": round(sum(r["input_tokens"] for r in ok) / len(ok))}


def run(decider: Decider, items: Sequence[Mapping[str, Any]], workers: int = 4) -> Dict[str, Any]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        rows = list(ex.map(lambda it: _one(decider, it), items))
    by = {t: _stats([r for r in rows if r["type"] == t]) for t in ("noul", "choice", "score")
          if any(r["type"] == t for r in rows)}
    return {"overall": _stats(rows), "by_type": by, "rows": rows}


def report(result: Dict[str, Any]) -> str:
    cols = ["n", "errors", "accuracy", "log_loss", "brier", "flip_rate", "p50_ms"]
    lines = ["         " + " ".join(f"{c:>9}" for c in cols)]
    for name, s in [("overall", result["overall"])] + list(result["by_type"].items()):
        lines.append(f"{name:<9}" + " ".join(f"{str(s.get(c, '')):>9}" for c in cols))
    misses = [r for r in result["rows"] if "error" in r or not r["correct"]]
    for r in misses:
        what = f"error: {r['error']}" if "error" in r else f"said {r['answer']!r} (p_gold {r['p_gold']:.2f})"
        lines.append(f"  miss [{r['type']}] {r['instructions'][:60]!r} want {r['gold']!r}, {what}")
    return "\n".join(lines)
