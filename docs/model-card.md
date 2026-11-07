---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
library_name: peft
language:
- en
- fr
- hi
- sw
- yo
- ha
pipeline_tag: text-generation
datasets:
- JustEddy1/kataa-bench
tags:
- agents
- tool-use
- ai-safety
- prompt-injection
- finance
- lora
- grpo
- agenttune
---

# Kataa: a 3B margin-desk agent that refuses unjustified actions

*Kataa* is [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global)
(3.35B, CC-BY-NC) fine-tuned with LoRA through Lexsi Labs' **AgentTune** to act as a margin-desk
agent: it reads a loan agreement, the position and two price oracles with tools, then records
exactly one decision: margin call, no action, or escalate to a human. It is meant to refuse when
the request is manipulated, the data is stale or contradictory, or the evidence isn't there.
Evaluated with Lexsi Labs' **AuditKIT**. Built for the Lexsi Labs AI Safety and Alignment
Hackathon, Track 3 (Agentic Fine-tuning).

> **It is not safe to let this model act autonomously.** On held-out attacks it takes an unsafe
> action on 25% of tasks (Mistral Large 4: 2.7%). Use it, if at all, as a cheap first-pass triage
> layer with a human or a stronger model behind it, and only for research.

**Collection:** [JustEddy1/kataa](https://huggingface.co/collections/JustEddy1/kataa) (model, benchmark, run artefacts, replay).

**Write-up:** the project [README](https://github.com/UEddy/Kataa#readme) (official), long form with every table in
[docs/article.md](https://github.com/UEddy/Kataa/blob/main/docs/article.md). Replay of recorded runs: [Space](https://huggingface.co/spaces/JustEddy1/kataa-demo).

Benchmark: [JustEddy1/kataa-bench](https://huggingface.co/datasets/JustEddy1/kataa-bench).
Code and every result table: [github.com/UEddy/Kataa](https://github.com/UEddy/Kataa).
Run artefacts (all jobs, adapters, logs, recorded evaluation runs):
[JustEddy1/kataa-runs](https://huggingface.co/JustEddy1/kataa-runs).

## Results (held-out attacks, greedy)

The held-out split uses two attack families never seen in training: instructions hidden in
structured data (JSON fields, markdown tables) and regulator/auditor/CEO authority personas.
95% bootstrap CIs over tasks; differences are paired.

| | Base Tiny Aya | **Kataa** |
|---|---|---|
| **Unsafe among tasks where it decided** | 61.0% [53.1, 68.9] | **25.0% [20.3, 30.0]** |
| Accuracy | 19.0% | 73.0% (+54.0 pp [+48.0, +60.0]) |
| Unsafe, all tasks | 29.7% | 25.0% (−4.7 pp [−11.3, +2.0], not significant) |
| Followed an attack | 21.0% | 12.3% (−8.7 pp [−14.3, −3.0]) |
| No decision | 51.3% | 0% |
| Evidence gathered before deciding | 0.3% | 97.3% |

Base Tiny Aya decides nothing on half the tasks, which makes its all-tasks unsafe rate look close
to Kataa's; when it does act, 61% of its actions are unsafe. On the in-distribution test split,
the warm start is *more* unsafe than base on the all-tasks rate (+7.0 pp [+0.3, +13.3]).

**Against a large hosted model**, on the same first 150 held-out tasks:

| | Kataa | Mistral Large 4 (open weights, via API, low reasoning effort) |
|---|---|---|
| Accuracy | 70.0% | 92.0% |
| Unsafe | 30.0% | 2.7% |
| Over-escalation | 0.0% | 2.7% |
| Cost per 1,000 checks | ~$0.32 (L40S list price, batched throughput) | $11.72 (list price) |

**Five languages, English-only training** (held-out set translated; tasks failing the translation
check excluded): accuracy 67–72% vs base 13–22%; the gap to English on the same tasks is −0.4 to
−2.3 pp, none significant.

**Where it fails:** stale prices on both oracles, oracle conflicts and active cure periods.
Accuracy on these is 0–9% for every trained model. It computes the price age or the oracle gap
correctly and then acts anyway.

## How it was trained

1. **Warm start (agentic distillation)**, `warmstart/`: scripted ideal runs for the 1,200 train
   tasks (fetch the agreement and position, fetch both oracles, compute price ages and gaps and the
   LTV with the tools, then decide), trained through AgentTune's `create_distill_trainer` with loss
   on the assistant turns only. LoRA r=16, alpha 32, all linear layers; 150 steps, lr 1e-4.
   Most of the improvement comes from this step.
2. **GRPO with a hardened reward (run 2)**, `run2/`: AgentTune `create_agentic_trainer("grpo")`
   with multi-turn tool rollouts, from the merged warm start. The reward credits a correct decision
   only after the evidence tools were called, penalises unsafe actions (−2) and over-escalation,
   rewards a parseable call list and penalises redundant calls. 90 steps × 8 rollouts, lr 5e-5,
   beta 0.04, temperature 0.9.

**Honest null result:** a run with a deliberately naive reward (run 1, same settings and code)
ends up statistically indistinguishable from run 2. Both beat the warm start on held-out by about
+5 pp accuracy and −3 pp unsafe.

**A repair that was rejected:** DPO on run 2's 284 unsafe train-split decisions (the correct
decision chosen over the unsafe one) cut the unsafe rate from 25.0% to 15.0% (−10.0 pp
[−16.7, −3.3]). The acceptance gate, fixed before the run, rejected it: the model had learned to
escalate 99% of tasks (accuracy 18.3%, over-escalation 81%). This repo does not contain that model.

## How to load it

The two adapters stack: merge `warmstart/` into Tiny Aya first, then `run2/`. Loading `run2/`
directly onto base Tiny Aya silently drops the warm start, so its config deliberately names a base
that cannot be loaded on its own.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE, REPO = "CohereLabs/tiny-aya-global", "JustEddy1/kataa"   # the base is gated: accept its licence
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, REPO, subfolder="warmstart").merge_and_unload()
model = PeftModel.from_pretrained(model, REPO, subfolder="run2").merge_and_unload()
```

`load_kataa.py` in this repo does the same. The model expects Kataa's protocol: the tool schemas in
the system prompt, a JSON call list at the end of each turn, and tool results returned as user
turns (see `kataa/protocol.py` and `kataa/eval/runner.py` in the code repo).

## Intended use and limits

- Research on agent safety, prompt-injection resistance and reward design. Not for real lending
  decisions, and not for autonomous action.
- One synthetic task family: margin checks on crypto-backed loans, with contracts written for the
  project and real daily prices.
- The tools are simulated; the model was never connected to real accounts.

## Licences

- Model weights (these adapters and anything merged from them): **CC-BY-NC 4.0**, inherited from
  [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global). Non-commercial use only.
- Trained with **AgentTune** and evaluated with **AuditKIT**, both by Lexsi Labs under their
  source-available non-commercial licences.
- Price data in the benchmark: © Coin Metrics, Inc., CC BY-NC 4.0.
