import json
import math
import re
import threading
import urllib.request

import pytest

from jevless import Choice, Decider, DecisionError, Noul, Score, fit_temperature
from jevless.server import make_server


class Fake:
    """A 'model' that prefers the option whose text contains `want`; `bias` adds a position preference for 'A'."""
    def __init__(self, want="", bias=0.0):
        self.want, self.bias, self.calls = want, bias, 0

    def first_token(self, prompt, top_k=20):
        self.calls += 1
        opts = re.findall(r"^([A-Z])\) (.*)$", prompt, re.M)
        tops = []
        for letter, text in opts:
            lp = (-0.1 if self.want and self.want in text else -3.0) + (self.bias if letter == "A" else 0.0)
            tops.append((letter, lp))
        return sorted(tops, key=lambda t: -t[1])[:top_k], len(prompt) // 4


def test_noul_choice_score():
    d = Decider(Fake("true:"))
    assert d.noul("s", "It holds.").noul > 0.9
    a = Decider(Fake("infra")).choice("503 since the deploy", "Which team?", {"billing": None, "infra": "servers", "sales": None})
    assert a.choice == "infra" and a.confidence > 0.8 and set(a.probabilities) == {"billing", "infra", "sales"}
    s = Decider(Fake("level 2")).score("s", "How relevant?", ["none", "some", "high"])
    assert 1.8 < s.score <= 2.0 and s.to_wire()["legend"]["2"] == "high"


def test_two_orders_cancel_position_bias():
    biased = Fake("", bias=2.0)                      # prefers whatever is shown first, knows nothing else
    one = Decider(biased, permutations=1).choice("s", "Pick", {"x": None, "y": None})
    two = Decider(biased, permutations=2).choice("s", "Pick", {"x": None, "y": None})
    assert one.probabilities["x"] > 0.8
    assert abs(two.probabilities["x"] - 0.5) < 1e-6 and two.flip


def test_limits_and_errors():
    with pytest.raises(DecisionError):
        Decider(Fake()).choice("s", "Pick", {str(i): None for i in range(27)})

    class Mute:
        def first_token(self, prompt, top_k=20):
            return [("I", -0.1), ("The", -0.2)], 10
    with pytest.raises(DecisionError, match="thinking"):
        Decider(Mute()).noul("s", "q")


def test_many_runs_in_parallel_and_keeps_keys():
    f = Fake("true:")
    out = Decider(f, workers=4).many("s", {f"q{i}": Noul(f"statement {i}") for i in range(8)})
    assert list(out) == [f"q{i}" for i in range(8)] and f.calls == 16


def test_fit_temperature_recovers_overconfidence():
    pairs = [([0.0, -6.0], 0)] * 7 + [([0.0, -6.0], 1)] * 3        # says 99.8% but is right 70% of the time
    t, before, after = fit_temperature(pairs)
    assert t > 1.5 and after < before


def test_wire_format_server_round_trip():
    srv = make_server(Decider(Fake("true:")), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1/systemone"
    body = {"state": {"ticket": "charged twice"}, "model": "jev-latest", "questions": {
        "urgent": {"type": "noul", "instructions": "This is urgent."},
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "true: ops": None}},
        "severity": {"type": "score", "instructions": "How severe?", "criteria": ["low", "true: high"]}}}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode())).read())
    assert set(r["answers"]) == {"urgent", "team", "severity"} and r["model"] == "jev-latest"
    assert r["answers"]["urgent"]["type"] == "noul" and 0 <= r["answers"]["urgent"]["noul"] <= 1
    assert r["answers"]["team"]["choice"] == "true: ops" and "confidence" in r["answers"]["team"]
    assert "legend" in r["answers"]["severity"] and r["usage"]["input_tokens"] > 0
    bad = urllib.request.Request(url, data=json.dumps({"state": "x", "questions": {"q": {"type": "maybe"}}}).encode())
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(bad)
    assert e.value.code == 422
    srv.shutdown()


def test_auto_reads_the_second_order_only_when_unsure():
    sure = Fake("true:")                               # confident: one reading
    a = Decider(sure, permutations="auto").noul("s", "It holds.")
    assert a.orders == 1 and sure.calls == 1 and a.noul > 0.9
    unsure = Fake("", bias=0.3)                       # weak preference for whatever is first: ask the reversed order
    b = Decider(unsure, permutations="auto").choice("s", "Pick", {"x": None, "y": None})
    assert b.orders == 2 and unsure.calls == 2 and abs(b.probabilities["x"] - 0.5) < 1e-6
    assert Decider(Fake("true:"), permutations=2).noul("s", "q").orders == 2
