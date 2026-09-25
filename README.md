# jevless

**Jev-style typed decisions from any model whose API exposes log-probabilities.**

An agent makes lots of small decisions: route this ticket, is this memory relevant, did that step succeed, how urgent is this. Asking a model to *write* "yes" and parsing it is slow and gives no probability. jevless asks the question once and reads the probabilities of the answer letters at the first output position. That's one prefill and one token, and you get a probability for every option.

It implements the three decision primitives popularised by TypeSafe's hosted **Jev** "System One" model:
- **Choice:** pick one of several options.
- **Noul:** how likely a statement is true.
- **Score:** a level on an ordered scale.

It includes a local server that speaks the same `/v1/systemone` request shape, but runs on models you choose: a local 9B in LM Studio, vLLM, llama.cpp, or OpenAI's API.

> Not affiliated with TypeSafe. "Jev" is their product; jevless is an independent, open implementation of the same idea over ordinary model APIs.

## Quick start

```python
from jevless import Decider, LMStudio, OpenAIChat

decide = Decider(LMStudio("qwen/qwen3.5-9b"))                 # or OpenAIChat("gpt-4.1-mini", api_key=...)

decide.noul("The API has returned 503 since the deploy.", "This is an outage.").noul          # 0.956
a = decide.choice("A customer writes: my card was charged twice.", "Which team should handle this?",
                  {"billing": None, "shipping": None, "security": None})
a.choice, a.probabilities, a.confidence, a.flip           # 'billing', {...}, 0.989, False
decide.score("Our checkout page has been down for all customers for the last hour.", "How urgent is this?",
             ["can wait", "this week", "today", "right now"]).score                    # 3.0 ("right now")

decide.many(state, {"urgent": Noul("..."), "team": Choice("...", {...})})    # several questions, in parallel
```

Or as a service:

```bash
pip install git+https://github.com/jbpayton/jevless
jevless probe --backend lmstudio --model qwen/qwen3.5-9b      # sanity-check a model before relying on it
jevless bench --model gpt-4.1-mini                             # 30 labelled decisions; key from $OPENAI_API_KEY
jevless serve --backend lmstudio --model qwen/qwen3.5-9b --port 8765

curl -s localhost:8765/v1/systemone -d '{
  "state": "A customer writes: my card was charged twice.",
  "questions": {
    "urgent": {"type": "noul", "instructions": "This needs a reply today."},
    "team":   {"type": "choice", "instructions": "Which team?", "criteria": {"billing": null, "shipping": null}},
    "tone":   {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "angry"]}}}'
```

The response has the same shape as Jev's: `{model, answers: {id: {type, noul | choice, probabilities, confidence | score, legend}}, usage}`. jevless adds `flip`.

(The numbers shown are from Qwen3.5-9B in LM Studio, as the model loaded there as `qwen35-9b`. The examples in `examples/` read the model name from `JEVLESS_MODEL`.)

## Any OpenAI-compatible endpoint

Point `--url` at the server and name the environment variable that holds the key:

```bash
export MY_LLM_KEY=...                                   # or whatever your system already sets
jevless probe --url https://llm.example.com/v1 --model my-model --api-key-env MY_LLM_KEY
jevless bench --url https://llm.example.com/v1 --model my-model --api-key-env MY_LLM_KEY
```

- **`--url`** can be the server root, a base URL ending in `/v1`, or a full `.../chat/completions` URL with its query string (Azure). It defaults to `$OPENAI_BASE_URL`, then `https://api.openai.com`.
- **The key** is read from `--api-key-env` (default `OPENAI_API_KEY`). `--api-key` also works, but anyone on the machine can see it in the process list.
- **`--key-header api-key`** sends the bare key in that header instead of `Authorization: Bearer` (Azure). `--header 'Name: value'` adds any other header.
- **Newer OpenAI models** that refuse `max_tokens` are retried with `max_completion_tokens` automatically.
- **Reasoning models** (OpenAI's o-series, for example) return no logprobs. Use a chat model such as `gpt-4.1-mini`, or switch thinking off (`--thinking off` on vLLM or llama-server).

## Comparing models

`jevless bench` runs 30 hand-written, labelled decisions (14 noul, 10 choice, 6 score) with a few traps: negation, sarcasm, a memory about the wrong person, a reply with nothing behind it, and pronouns that need the rest of the sentence. It reports accuracy, log loss, Brier score, how often the two option orders disagree, and latency, then lists every miss. `--file` takes your own decisions as JSONL (the format is in `jevless/bench.py`) and `--out` saves every decision.

| Model | Served by | Accuracy | Noul | Choice | Score | Log loss | Flip rate |
|---|---|---|---|---|---|---|---|
| Qwen3.5-9B (Q4) | LM Studio | 0.933 | 14/14 | 10/10 | 4/6 | 0.151 | 0.03 |
| Qwen3.5-0.8B (Q8) | llama-server, CPU, via `--api-key-env` | 0.500 | 5/14 | 8/10 | 2/6 | 0.902 | 0.40 |

The 0.8B answers "true" to nearly every false statement. That is the failure `probe` flags, and why a small model needs checking before you trust it. The 9B's two misses are middle-of-scale scores ("this week" read as "today", "frustrating" as "angry"). Thirty items is a smoke test for comparing models on the same footing, not a benchmark result.

## How it works

1. **The prompt.** The state goes first, then the question, then the options as single letters: `A) billing`, `B) shipping`… With the state first, servers with prefix caching reuse it across a fan-out of questions.
2. **The readout.** The model makes one forward pass. jevless reads the log-probabilities of the option letters at the first output position and renormalises over the letters on offer.
3. **Both orders.** By default the options are also read in reverse order and the two readings are averaged, which cancels the model's preference for an option's *position*. `flip` is true when the two readings disagree: a cheap "look closer" signal.
4. **Calibration.** Raw probabilities are uncalibrated. Log decisions (`Decider(..., log=callback)`), record what turned out to be right, and fit a temperature per question type (`fit_temperature`) once you have about 50 labels.

## Backends

| Backend | Works with | Notes |
|---|---|---|
| `OpenAIChat` | OpenAI (non-reasoning chat models), vLLM, llama.cpp's llama-server, SGLang, other OpenAI-compatible servers that return `logprobs` | `thinking="off"` disables Qwen3-style thinking on vLLM or llama-server. The top-k window is at most 20 |
| `LMStudio` | LM Studio | Its chat endpoint returns no logprobs; `/v1/responses` does, with reasoning switched off |
| `LlamaCpp` | llama.cpp's native `/completion` | Raw prompt, so pass the model's chat template |
| `Transformers` | Any local Hugging Face causal LM | In-process logits. Needs `torch` and `transformers` |

**Not supported:** APIs that don't expose token log-probabilities (for example Anthropic's), and reasoning models that must think before answering.

## What it's good for, measured

These numbers come from building [hermes-sophia](https://github.com/jbpayton/hermes-sophia), a memory system that uses this technique for all its decisions. They were measured on one machine (2× RTX 3090, LM Studio). The research scripts and data are in that repo.

- **Decisions that similarity can't make.** Deciding whether recalled memories actually answer a question: cosine similarity couldn't separate answers from near-misses (0.80 against 0.78). A Qwen3.5-9B readout got 32 of 34 right, and never missed an answer that was there.
- **Navigation.** On room-by-room choices in a memory graph ("which door leads to the answer?"), the 9B and a 27B readout got every choice right. Small purpose-built decision encoders (Laya 421M, Von 395M) were fast but wrong without fine-tuning.
- **Speed.** One question on the 9B takes 210–640 ms cold (depending on state length) and 110–150 ms with the state cached. Eight in flight gave 6.6 decisions per second.
- **Wording and both orders matter.**
  - Checking whether a reply invented facts: one negatively worded question in one order didn't separate true from false cases at all. A positive wording read in both orders did (0.59+ against 0.34 or less).
  - Deciding whether a new fact replaces an old one: false cases topped out at 0.74 and real changes started at 0.92. The fix was the threshold, not the model.
  - So measure on a few labelled cases before trusting a threshold.
- **Model size matters.** A Qwen3.5-0.8B waved an off-topic question through a relevance check at 0.90, where the 9B said 0.45. `jevless probe` catches this kind of model before you rely on it.

## Limits

jevless is a thin approximation of Jev: an ordinary model and a readout, with no purpose-trained decision head and no calibration out of the box. For routing, gating and triage with a decent model that is often good enough. At worst, it shows how the technique works in about 500 lines of Python.

- **Up to 26 options per question** (single letters); Jev allows 255.
- **Probabilities are uncalibrated** until you fit temperatures from your own labelled decisions.
- **Reading both orders doubles the calls.** Use `permutations=1` when speed matters more than position bias.
- **The model must answer immediately:** switch reasoning or thinking off.

## Related work

- **TypeSafe's Jev:** hosted, proprietary, and the origin of the Choice / Noul / Score framing and the wire format.
- **Open readout projects:** jobe, reflex and SemIf read decisions from frozen open models in-process. jevless applies the same idea over the HTTP APIs of whatever server you already run.

## License

MIT
