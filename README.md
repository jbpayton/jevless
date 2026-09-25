# jevless

**Jev-style typed decisions from any model whose API exposes log-probabilities.**

An agent makes lots of small decisions: route this ticket, is this memory relevant, did that step succeed, how urgent is this. Asking a model to *write* "yes" and parsing it is slow and gives no probability. jevless asks the question once and reads the probabilities of the answer letters at the first output position. That's one prefill and one token, and you get a probability for every option.

It implements the three decision primitives popularised by TypeSafe's hosted **Jev** "System One" model:
- **Choice:** pick one of several options.
- **Noul:** how likely a statement is true.
- **Score:** a level on an ordered scale.

It includes a local server that speaks the same `/v1/systemone` request shape, but runs on models you choose: a local 9B in LM Studio, vLLM, llama.cpp, or OpenAI's API.

> Not affiliated with TypeSafe. "Jev" is their product; jevless is an independent, open implementation of the same idea over ordinary model APIs.

## Results: 32 models, one decision at a time

The same 60 labelled decisions (the `basic` and `hard` sets below) put to every model two API keys could reach, plus a local 9B, on 25 September 2026. **Round trip** is the median time for one request from this machine, network included. A **decision** is two requests, one per option order, made one after the other.

| Model | Provider | How it decides | Basic | Hard | Round trip | Per decision |
|---|---|---|---|---|---|---|
| gpt-5.5 | OpenAI | readout, reasoning off | 1.00 | **1.00** | 0.87 s | 1.7 s |
| gpt-6-sol | OpenAI | readout, reasoning off | 1.00 | **1.00** | 1.24 s | 2.5 s |
| claude-opus-4-6 | Anthropic | answer (may reason) | 1.00 | **1.00** | 1.34 s | 2.7 s |
| claude-opus-5-5 | Anthropic | answer (may reason) | 1.00 | **1.00** | 2.00 s | 4.0 s |
| claude-fable-5 | Anthropic | answer (may reason) | 1.00 | **1.00** | 2.57 s | 5.1 s |
| claude-fable-5-1 | Anthropic | answer (may reason) | 1.00 | **1.00** | 3.52 s | 7.0 s |
| gpt-5.4 | OpenAI | readout, reasoning off | 1.00 | **0.97** | 0.73 s | 1.5 s |
| claude-sonnet-4-6 | Anthropic | answer (may reason) | 1.00 | **0.97** | 1.02 s | 2.0 s |
| claude-opus-4-5-20251101 | Anthropic | answer (may reason) | 1.00 | **0.97** | 1.11 s | 2.2 s |
| claude-sonnet-4-5-20250929 | Anthropic | answer (may reason) | 1.00 | **0.97** | 2.47 s | 4.9 s |
| gpt-4.1 | OpenAI | readout | 1.00 | **0.93** | 0.58 s | 1.2 s |
| gpt-4.1-mini | OpenAI | readout | 1.00 | **0.93** | 0.60 s | 1.2 s |
| gpt-5.4-mini | OpenAI | readout, reasoning off | 1.00 | **0.93** | 0.62 s | 1.2 s |
| gpt-5.1 | OpenAI | readout, reasoning off | 1.00 | **0.93** | 0.80 s | 1.6 s |
| gpt-5.6-sol | OpenAI | readout, reasoning off | 1.00 | **0.93** | 0.91 s | 1.8 s |
| claude-opus-4-8 | Anthropic | answer (may reason) | 1.00 | **0.93** | 0.92 s | 1.9 s |
| claude-sonnet-5 | Anthropic | answer (may reason) | 1.00 | **0.93** | 1.05 s | 2.1 s |
| gpt-4o | OpenAI | readout | 1.00 | **0.90** | 0.57 s | 1.1 s |
| claude-opus-4-7 | Anthropic | answer (may reason) | 1.00 | **0.90** | 0.75 s | 1.5 s |
| gpt-5.6-luna | OpenAI | readout, reasoning off | 0.97 | **0.90** | 0.77 s | 1.5 s |
| gpt-5.6-terra | OpenAI | readout, reasoning off | 1.00 | **0.90** | 0.88 s | 1.8 s |
| gpt-6-luna | OpenAI | readout, reasoning off | 1.00 | **0.90** | 0.89 s | 1.8 s |
| gpt-5.2 | OpenAI | readout, reasoning off | 1.00 | **0.90** | 0.93 s | 1.9 s |
| gpt-4-turbo | OpenAI | readout | 1.00 | **0.90** | 1.18 s | 2.4 s |
| claude-haiku-4-5-20251001 | Anthropic | answer (may reason) | 0.97 | **0.87** | 0.53 s | 1.1 s |
| gpt-4 | OpenAI | readout | 1.00 | **0.87** | 0.84 s | 1.7 s |
| Qwen3.5-9B Q4 (this machine, LM Studio) | local | readout | 0.93 | **0.87** | 1.30 s | 2.6 s |
| claude-opus-5 ¹ | Anthropic | answer (may reason) | 0.93 | **0.87** | 1.56 s | 3.1 s |
| gpt-4.1-nano | OpenAI | readout | 0.93 | **0.77** | 0.50 s | 1.0 s |
| gpt-4o-mini | OpenAI | readout | 1.00 | **0.73** | 0.55 s | 1.1 s |
| gpt-5.4-nano | OpenAI | readout, reasoning off | 0.90 | **0.67** | 0.66 s | 1.3 s |
| gpt-3.5-turbo | OpenAI | readout | 0.87 | **0.67** | 0.89 s | 1.8 s |

¹ claude-opus-5 left 5 of its 60 replies without an answer letter (empty, or reasoning written out as text). They count as wrong.

**How to read it:**
- **Two ways of deciding.** *Readout* models answer in one token, and jevless reads the probability of every option; that is what jevless is for. Claude models have no logprobs API, so they run in *answer* mode: the model replies and jevless takes the letter. claude-opus-5-5, claude-fable-5 and claude-fable-5-1 can't switch thinking off, so they ran with adaptive thinking at low effort. The other nine ran with thinking disabled, though some still work through a hard item in the reply itself (claude-sonnet-4-5 does). Reasoning helps on the hard set and costs time. Compare accuracy within a mode; the round-trip column compares across them.
- **Fastest:** gpt-4.1-nano (0.50 s), claude-haiku-4-5 (0.53 s), gpt-4o (0.57 s), gpt-4.1 and gpt-4.1-mini (about 0.6 s).
- **Most accurate for the time:** gpt-4.1 and gpt-4.1-mini (0.93 at 0.6 s), gpt-5.4 (0.97 at 0.73 s), gpt-5.5 (1.00 at 0.87 s). In answer mode: claude-opus-4-6 (1.00 at 1.3 s) and claude-sonnet-4-6 (0.97 at 1.0 s).
- **Small models slip on the hard set:** gpt-4o-mini 0.73, gpt-4.1-nano 0.77, gpt-5.4-nano 0.67.
- **The local 9B** was timed while this machine's GPUs were busy with other benchmarks. Idle, one request on it takes 0.2–0.6 s, and 0.1–0.15 s when the state is already cached (see [measured](#what-its-good-for-measured)).
- **Halving the time.** `--orders 1` makes one request per decision, but gives up the protection against position bias. Several questions about the same state (`many`) run in parallel.
- **Scale.** Thirty items per set: fine for comparing models on the same footing, not a benchmark result.

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
jevless bench --model gpt-4.1-mini --set hard                  # labelled decisions; key from $OPENAI_API_KEY
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
- **Reasoning models** need reasoning off before they return logprobs: `--reasoning-effort none` for OpenAI's GPT-5.1 and later (set automatically when refused), `--thinking off` on vLLM or llama-server. Some never return them (OpenAI's o-series, gpt-5, gpt-5-mini, gpt-5-nano).

## Comparing models

`jevless bench` runs hand-written, labelled decisions (noul, choice and score) and reports accuracy, log loss, Brier score, how often the two option orders disagree, and latency, then lists every miss.
- **`--set basic`** (the default): 30 everyday decisions with a few traps (negation, sarcasm, a memory about the wrong person, a reply with nothing behind it, Winograd pronouns). Frontier models get all of it right.
- **`--set hard`**: 30 more, harder for a model that has to answer in one token with no reasoning. They include arithmetic, a leap-year due date, stacked negation, "all except", log lines out of order, a web page with a hidden instruction, units, a syllogism with a false premise, a code trace, a false-belief question, a notice deadline, and a right answer shown as option 7 of 8.
- **`--file`** takes your own decisions as JSONL (the format is in `jevless/bench.py`); `--out` saves every decision.

Results are [at the top](#results-32-models-one-decision-at-a-time). Notes on running them:

- **Which OpenAI models work.** The GPT-3.5, GPT-4, GPT-4o and GPT-4.1 families return logprobs as they are. GPT-5.1 and later (including gpt-6-luna and gpt-6-sol) return them only with reasoning off, which jevless sets on its own. gpt-5, gpt-5-mini, gpt-5-nano, the o-series and gpt-6-astra refuse logprobs; the `-chat-latest` aliases are retired.
- **Adapting to newer models.** jevless reads each refusal and adjusts once: `max_completion_tokens` instead of `max_tokens`, with room for a few hidden tokens; `top_logprobs` capped at 5; no `temperature`; reasoning off. The Anthropic backend does the same for `temperature` and thinking.
- **Log loss for gpt-5.2 and later has a floor.** These models list only alternatives with real probability, so an option they leave out gets jevless's floor. That caps how confident a readout can look, so their log loss can't go much below about 0.01. Their accuracy is exact. Answer-mode backends report no log loss at all.
- **What the hard set exposes.** Among readout models, the notice-deadline item ("renews unless cancelled at least 30 days before October 1; cancelled September 5") was missed by 17 of 20. Unit prices, distinct names on a sign-in sheet and the leap-year date each caught several. These are the decisions to hand to a reasoning step rather than a one-token readout.
- **One judgment call.** Whether "password reset emails aren't being sent" is *major* or *critical* splits older models (critical) from newer ones (major, the label here).

A small local model, for contrast: Qwen3.5-0.8B through llama-server scored 0.50 on the basic set. It answers "true" to nearly every false statement, which is what `probe` is there to catch.

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
| `Anthropic` | Claude models | No logprobs: answer mode (the model's letter, no probabilities). Thinking off where allowed, else lowest effort |

**No logprobs, answers only:** Anthropic's API returns no log-probabilities. `--backend anthropic` (key from `$ANTHROPIC_API_KEY`) still runs decisions in *answer* mode: the model replies, jevless takes the letter, and there are no probabilities or calibration.

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
