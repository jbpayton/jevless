"""Typed decisions from a language model's first-token probabilities: one prefill, no generated text.

Every option is shown as a single letter. The probability of each letter at the first output position,
renormalized over the letters on offer, is the answer. Reading twice with the options reversed and averaging
cancels position bias; ``flip`` says whether the two readings disagreed -- a cheap "look closer" signal.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
Text = Union[str, Mapping[str, Any], Sequence[Any]]


class DecisionError(RuntimeError):
    pass


# ------------------------------------------------------------------ questions
@dataclass(frozen=True)
class Noul:
    """How likely is it that the statement holds? (Jev's 'noul': a calibrated yes/no.)"""
    instructions: Text
    criteria: Optional[Mapping[str, str]] = None          # optional descriptions for "true" and "false"


@dataclass(frozen=True)
class Choice:
    """Which one option fits best? criteria maps option keys to descriptions (or None)."""
    instructions: Text
    criteria: Mapping[str, Optional[str]]


@dataclass(frozen=True)
class Score:
    """Where on an ordered scale? criteria lists 2-10 level descriptions, lowest first."""
    instructions: Text
    criteria: Sequence[str]


Question = Union[Noul, Choice, Score]


def question_from_wire(q: Mapping[str, Any]) -> Question:
    t = q.get("type")
    if t == "noul":
        return Noul(q["instructions"], q.get("criteria"))
    if t == "choice":
        c = q.get("criteria")
        if isinstance(c, (list, tuple)):
            c = {str(x): None for x in c}
        if not isinstance(c, Mapping) or not c:
            raise DecisionError("choice needs criteria: a map of option -> description")
        return Choice(q["instructions"], c)
    if t == "score":
        c = q.get("criteria")
        if not isinstance(c, (list, tuple)) or not 2 <= len(c) <= 10:
            raise DecisionError("score needs criteria: a list of 2-10 ordered levels")
        return Score(q["instructions"], list(c))
    raise DecisionError(f"unknown question type {t!r} (noul | choice | score)")


# -------------------------------------------------------------------- answers
@dataclass
class Answer:
    type: str
    keys: List[str]
    probabilities: Dict[str, float]
    flip: bool = False
    latency_ms: float = 0.0
    input_tokens: int = 0
    decision_id: str = ""
    legend: Dict[str, str] = field(default_factory=dict)
    raw: List[Dict[str, float]] = field(default_factory=list)

    @property
    def choice(self) -> str:
        return max(self.probabilities, key=self.probabilities.get)

    @property
    def noul(self) -> float:
        return self.probabilities.get("true", 0.0)

    @property
    def score(self) -> float:
        """Expected level (0 = lowest)."""
        return sum(int(k) * p for k, p in self.probabilities.items())

    @property
    def confidence(self) -> float:
        """How far the top option is above uniform, from 0 (no preference) to 1 (certain)."""
        k = len(self.probabilities)
        p = max(self.probabilities.values())
        return (p - 1.0 / k) / (1.0 - 1.0 / k) if k > 1 else 1.0

    def to_wire(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"type": self.type}
        if self.type == "noul":
            out["noul"] = round(self.noul, 6)
        else:
            out["probabilities"] = {k: round(v, 6) for k, v in self.probabilities.items()}
            out["confidence"] = round(self.confidence, 6)
            if self.type == "choice":
                out["choice"] = self.choice
            else:
                out["score"] = round(self.score, 6)
                out["legend"] = self.legend
        out["flip"] = self.flip
        return out


# ------------------------------------------------------------------- prompting
def _text(x: Text) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=1)


def _options(q: Question):
    if isinstance(q, Choice):
        keys = [str(k) for k in q.criteria]
        return keys, [k if not q.criteria[k] else f"{k}: {q.criteria[k]}" for k in keys], "choice"
    if isinstance(q, Noul):
        c = q.criteria or {}
        return ["false", "true"], [f"false: {c.get('false', 'no, the statement does not hold')}",
                                   f"true: {c.get('true', 'yes, the statement holds')}"], "noul"
    if isinstance(q, Score):
        return [str(i) for i in range(len(q.criteria))], [f"level {i}: {d}" for i, d in enumerate(q.criteria)], "score"
    raise DecisionError(f"unknown question {q!r}")


TAILS = {"choice": "Choose the single best option.", "noul": "Decide whether the statement is true of the STATE.",
         "score": "Rate the STATE on the ordered levels."}


def render(state_text: str, instructions: str, descs: Sequence[str], qtype: str) -> str:
    """State first, question last: a server with prefix caching reuses the state across a fan-out."""
    opts = "\n".join(f"{LETTERS[i]}) {d}" for i, d in enumerate(descs))
    return (f"STATE:\n{state_text}\n\nQUESTION: {instructions}\n{TAILS[qtype]}\nOptions:\n{opts}\n"
            f"Answer with the single letter only.")


def _softmax(v: Sequence[float]) -> List[float]:
    m = max(v)
    w = [math.exp(x - m) for x in v]
    s = sum(w)
    return [x / s for x in w]


# ---------------------------------------------------------------------- decider
class Decider:
    """decide = Decider(LMStudio("qwen/qwen3.5-9b")); decide.noul(state, "The customer is asking for a refund.")"""

    def __init__(self, backend, *, permutations: int = 2, temperatures: Optional[Dict[str, float]] = None,
                 top_k: int = 20, workers: int = 4, log: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.backend = backend
        self.permutations = max(1, min(2, permutations))
        self.temperatures = {"choice": 1.0, "noul": 1.0, "score": 1.0, **(temperatures or {})}
        self.top_k, self.workers, self.log = top_k, workers, log

    def ask(self, state: Text, q: Question, permutations: Optional[int] = None) -> Answer:
        keys, descs, qtype = _options(q)
        k = len(keys)
        if not 2 <= k <= len(LETTERS):
            raise DecisionError(f"jevless supports 2..{len(LETTERS)} options per question (got {k})")
        perms = self.permutations if permutations is None else max(1, min(2, permutations))
        orders = [list(range(k))] + ([list(range(k))[::-1]] if perms == 2 else [])
        state_text, instructions = _text(state), _text(q.instructions)
        t0 = time.perf_counter()
        per_order, raws, tokens = [], [], 0
        for order in orders:
            tops, n_in = self.backend.first_token(render(state_text, instructions, [descs[i] for i in order], qtype),
                                                  max(self.top_k, k))
            tokens += n_in
            letter_lp: Dict[str, float] = {}
            for tok, lp in tops:
                t = tok.strip().rstrip(")").strip()
                if len(t) == 1 and t in LETTERS[:k]:
                    letter_lp[t] = max(letter_lp.get(t, -1e9), lp)
            if not letter_lp:
                raise DecisionError(f"no option letter among the model's top tokens {[t for t, _ in tops][:6]} "
                                    "(is reasoning/thinking switched off?)")
            raws.append(letter_lp)
            floor = min(letter_lp.values()) - 5.0            # a letter outside the top-k: well below the worst seen
            probs = _softmax([letter_lp.get(LETTERS[j], floor) / self.temperatures[qtype] for j in range(k)])
            declared = [0.0] * k
            for j, oi in enumerate(order):
                declared[oi] = probs[j]
            per_order.append(declared)
        probs = [sum(p[i] for p in per_order) / len(per_order) for i in range(k)]
        flip = len(per_order) > 1 and max(range(k), key=lambda i: per_order[0][i]) != max(range(k), key=lambda i: per_order[1][i])
        ans = Answer(qtype, keys, {keys[i]: probs[i] for i in range(k)}, flip, (time.perf_counter() - t0) * 1000,
                     tokens, raw=raws, legend={str(i): d for i, d in enumerate(q.criteria)} if qtype == "score" else {})
        if self.log:
            rec = {"ts": time.time(), "type": qtype, "state_sha": hashlib.sha1(state_text.encode()).hexdigest()[:16],
                   "instructions": instructions, "options": keys, "probabilities": probs, "raw": raws, "flip": int(flip)}
            rec["id"] = hashlib.sha1(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()[:16]
            ans.decision_id = rec["id"]
            try:
                self.log(rec)
            except Exception:
                pass
        return ans

    def noul(self, state: Text, instructions: Text, criteria: Optional[Mapping[str, str]] = None, **kw) -> Answer:
        return self.ask(state, Noul(instructions, criteria), **kw)

    def choice(self, state: Text, instructions: Text, criteria: Mapping[str, Optional[str]], **kw) -> Answer:
        return self.ask(state, Choice(instructions, criteria), **kw)

    def score(self, state: Text, instructions: Text, criteria: Sequence[str], **kw) -> Answer:
        return self.ask(state, Score(instructions, criteria), **kw)

    def many(self, state: Text, questions: Mapping[str, Question]) -> Dict[str, Answer]:
        """Several questions about one state, in parallel (the shared state prefix is cached by most servers)."""
        items = list(questions.items())
        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as ex:
            answers = list(ex.map(lambda kv: self.ask(state, kv[1]), items))
        return {k: a for (k, _), a in zip(items, answers)}

    def system_one(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        """A Jev-style request -- {state, model, questions: {id: {type, instructions, criteria}}} -- answered with
        {model, answers: {id: {...}}, usage}."""
        if "state" not in request or not isinstance(request.get("questions"), Mapping) or not request["questions"]:
            raise DecisionError("the request needs 'state' and a non-empty 'questions' map")
        qs = {qid: question_from_wire(q) for qid, q in request["questions"].items()}
        answers = self.many(request["state"], qs)
        return {"model": request.get("model", "jevless"), "answers": {k: a.to_wire() for k, a in answers.items()},
                "usage": {"input_tokens": sum(a.input_tokens for a in answers.values()),
                          "output_tokens": sum(self.permutations for _ in answers)}}


# ------------------------------------------------------------------ calibration
def fit_temperature(pairs: Sequence[tuple]) -> tuple:
    """Temperature that minimizes negative log-likelihood. pairs = [(letter_logprobs_in_option_order, gold_index)].
    Returns (temperature, nll_at_1, nll_at_temperature)."""
    def nll(T):
        tot = 0.0
        for lps, g in pairs:
            p = _softmax([x / T for x in lps])
            tot -= math.log(max(p[g], 1e-12))
        return tot / len(pairs)
    lo, hi = 0.2, 5.0
    phi = (math.sqrt(5) - 1) / 2
    a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
    for _ in range(50):
        if nll(a) < nll(b):
            hi, b, a = b, a, hi - phi * (hi - lo)
        else:
            lo, a, b = a, b, lo + phi * (hi - lo)
    t = round((lo + hi) / 2, 3)
    return t, nll(1.0), nll(t)
