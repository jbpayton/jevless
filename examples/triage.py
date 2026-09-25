"""Route support tickets and flag urgent ones, with probabilities, using a local model."""
import os

from jevless import Choice, Decider, LMStudio, Noul, Score

decide = Decider(LMStudio(os.environ.get("JEVLESS_MODEL", "qwen/qwen3.5-9b")))
tickets = ["My card was charged twice for the same order.",
           "The tracking page says delivered but nothing arrived.",
           "Someone logged into my account from another country!"]
for t in tickets:
    a = decide.many(t, {"team": Choice("Which team should handle this?", {"billing": None, "shipping": None, "security": None}),
                        "urgent": Noul("This needs a human reply today."),
                        "mood": Score("How upset is the customer?", ["calm", "annoyed", "angry"])})
    print(f"{a['team'].choice:9s} urgent={a['urgent'].noul:.2f} mood={a['mood'].score:.1f} flip={a['team'].flip} | {t}")
