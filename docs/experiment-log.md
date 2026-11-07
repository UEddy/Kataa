# Experiment log

All GPU work runs on HF Jobs. Artifacts are in `JustEddy1/kataa-runs` (private until submission), and costs are HF Jobs running time at list price.

## Nov 6 (Nov 9 schedule items, run early)

| Job | Hardware | What | Result | Cost |
|---|---|---|---|---|
| 6aed728c… | t4-small | Smoke: AgentTune's Tiny Aya README example | Out of memory: 262K-vocab logits do not fit in 16 GB | ~$0.03 |
| 6aed73cb… | l40sx1 | Smoke: README example, then our warm start and GRPO (2 steps) | All passed; ~45 s per rollout with transformers generation | ~$0.40 |
| 6aed78a9… | l40sx1 | Smoke: GRPO with colocated vLLM 0.22.1 | ~5 s per rollout (~9x faster); peak 38.3 GiB | ~$0.13 |
| 6aed8894… | l40sx1 | Warm start: 1,200 ideal runs, 1 epoch, 150 steps, assistant-only loss | Loss 1.23 to 0.08, plateau by step ~50; checkpoints every 25 steps | ~$0.55 |
| 6aedb555… | l40sx1 | Step 0, attempt 1 | vLLM 0.22.1 LoRA crashes on Tiny Aya; switched to merged models | ~$0.10 |
| 6aedbe20… | l40sx1 | Step 0, attempt 2 | FlashInfer sampler JIT needs nvcc; set `VLLM_USE_FLASHINFER_SAMPLER=0` | ~$0.19 |
| 6aedc1a4… | l40sx1 | Step 0: 40 val tasks x 4 samples, temperature 0.9 | Table below | ~$0.31 |

### Step 0: choosing run 1's starting checkpoint

The rule was fixed before seeing the data: start from the most-trained checkpoint whose naive reward varies within at least **30%** of the 4-sample task groups. At that rate about half of all GRPO steps, with 2 groups each, carry a learning signal.

| Metric | Base | WS-25 | WS-75 | WS-150 |
|---|---|---|---|---|
| Accuracy | 10.0% | 25.0% | 37.5% | 43.1% |
| Naive-reward spread (groups) | 30.0% | 62.5% | 70.0% | **75.0%** |
| Full evidence before deciding | 0.0% | 53.1% | 95.6% | 95.0% |
| Format OK | 30.6% | 56.3% | 98.8% | 98.1% |
| No decision | 63.1% | 41.9% | 1.3% | 1.3% |
| Unsafe | 25.6% | 20.6% | 31.9% | 30.6% |
| Over-escalation | 8.8% | 7.5% | 16.3% | 6.9% |
| Followed an attack | 15.6% | 6.9% | 10.6% | 11.3% |
| Mean hardened reward | -1.11 | -0.62 | -0.25 | -0.15 |

The warm start learned the protocol (evidence, format, always deciding) but not the judgment: accuracy is 43% and 31% of decisions are unsafe. **Decision: run 1 starts from WS-150** (`warmstart/4ef6d074ec4d`).

## Run 1: naive reward (job 6aedc974…, plus two failed attempts)

- Attempt 6aedc458 ran out of memory in backward with 4 rollouts per micro-batch. Now 1 per micro-batch, accumulated over 8. (~$0.12)
- Attempt 6aedc601 trained with grad_norm ~1e-16 and was cancelled after 10 steps. Cause: TRL's vLLM importance-sampling correction. AgentTune scores the model's turns as one concatenated sequence, while vLLM sampled each turn from the rendered chat, so the per-sequence ratio collapses. Correction now off, which matches the transformers-generation path. (~$0.36)
- The completed run: 120 steps x 8 rollouts, lr 1e-5, beta 0.04, temperature 0.9, from WS-150. 1.73 h, ~$3.12. KL stayed below ~0.0012.

Training log (20-step windows): reward 0.45 / 0.53 / 0.41 / 0.42 / 0.49 / 0.55. Evidence-before-decision stayed 97-100%; unsafe 30-33%; escalate 13-18%; calls 5.6-6.5.

Held-out validation. An evaluation that merged run 1's adapter onto base Tiny Aya, dropping the warm start, gave meaningless numbers (eval/run1/8c050d7f4788, run-1 rows only). It was redone as warm start + run 1 (eval/run1-fixed/90eda24675c5):

| Greedy, val (150) | WS-150 | Run 1 |
|---|---|---|
| Accuracy | 42.7% | 43.3% |
| Evidence first | 97.3% | 98.0% |
| Unsafe | 40.7% | 39.3% |
| Followed attack | 13.3% | 12.0% |
| Over-escalation | 0.7% | 0.0% |

**Verdict:** no reward gaming within 120 steps, and no learning either; the policy barely moved (KL ~0.001).

### Why the warm start is stuck at ~43% (WS-150, greedy val)

| Condition | n | Accuracy | Main error |
|---|---|---|---|
| clean | 67 | 54% | margin call predicted as no action (20), and the reverse (10) |
| cure_active | 20 | 45% | calls during a running cure period (10) |
| cure_expired | 11 | 36% | no action when a call was due (7) |
| missing_position | 5 | 100% | — |
| oracle_conflict | 14 | 0% | never escalates; the gap is computed correctly but the rule is not applied |
| oracle_within | 15 | 67% | — |
| stale_both | 13 | 0% | never escalates; price ages miscalculated (50 h reported as "2.0h old") |
| stale_primary | 5 | 0% | — |

Two failure types:

1. **Arithmetic.** Multi-digit products are pattern-matched, not computed: the same "$2,504,256.36" collateral value appears for two different loans. Ages across day boundaries are wrong too. RL cannot add arithmetic the model lacks.
2. **Judgment.** The divergence is computed correctly (+14.36% vs a 3% limit) and then ignored. The hardened reward targets exactly this.

## Calculator and hours_between, warm start 2 (WS2), arithmetic check

- Added AgentTune's `calculator` behind a strict numeric gate, and `hours_between`. The ideal runs now compute ages, gaps, collateral value and LTV with them (5 tool turns).
- WS2 (job 6aedf7e0…, ~$0.79): 150 steps, loss 0.93 to 0.005, conversations up to 4,977 tokens.
- Arithmetic check (job 6aee040c…, ~$0.20), greedy on the full val split:

| | WS1 | WS2 |
|---|---|---|
| Accuracy | 42.7% | 66.0% |
| Unsafe | 40.7% | 31.3% |
| Followed an attack | 13.3% | 7.3% |
| Evidence first | 97.3% | 99.3% |
| Price ages correct (hours_between) | — | 100% (387/387) |
| LTV computed with the calculator | — | 98.3% |
| ... of those, within 0.5 pp of truth | — | 90.5% |

WS2 accuracy by condition (greedy): clean 96%, cure_expired 91%, oracle_within 87%, missing_position 100%, stale_primary 40%, **cure_active 15%, oracle_conflict 14%, stale_both 0%**.

The arithmetic is fixed. The remaining errors are judgment: the model computes that a price is 50 h old against a 24 h limit, then acts anyway. Naive-reward spread (40 x 4, temperature 0.9) is 67.5%, so run 1 may start from WS2.

## Runs 1 and 2 (lr 5e-5, 90 steps, from WS2)

- Run 1, naive reward: job 6aee0a3a…, ~$3.90. Passed `check_run.py` (max KL 0.021).
- Run 2, hardened reward: job 6aee40ea…, ~$3.81. Same code snapshot (169140127fca) and settings; only the reward differs. Passed `check_run.py` (max KL 0.105).
- Training logs show no reward gaming in run 1: evidence-before-decision stayed at 97-100% and the decision mix was stable.

## Final evaluation (job 6aee9bf3…, ~$1.67, after ~4 h queued)

Greedy. Translated rows are before the exclusion rule in `data/translations/exclusions.json` is applied. Results are in `kataa-runs/eval/final/88e8bac4dc62/`.

| Held-out (300) | Base | Prompted | WS2 | Run 1 | Run 2 |
|---|---|---|---|---|---|
| Accuracy | 19.0% | 20.7% | 68.3% | 73.7% | 73.0% |
| Unsafe | 29.7% | 25.3% | 28.0% | 24.3% | 25.0% |
| Followed an attack | 21.0% | 20.0% | 14.7% | 12.0% | 12.3% |
| Over-escalation | 4.7% | 16.3% | 1.3% | 0.0% | 0.3% |
| Evidence first | 0.3% | 5.7% | 97.7% | 98.7% | 97.3% |
| No decision | 51.3% | 44.7% | 0% | 0% | 0% |

| Test (300) | Base | Prompted | WS2 | Run 1 | Run 2 |
|---|---|---|---|---|---|
| Accuracy | 13.0% | 16.0% | 66.3% | 68.3% | 67.7% |
| Unsafe | 22.7% | 24.0% | 29.7% | 29.0% | 29.3% |
| Followed an attack | 7.7% | 8.7% | 3.7% | 3.7% | 3.7% |

Translated held-out (base vs run 2), accuracy / unsafe:

- French (300): 21.3% / 31.7% vs 72.7% / 25.3%
- Hindi (150): 20.0% / 24.0% vs 69.3% / 30.0%
- Swahili (150): 16.0% / 26.0% vs 68.7% / 30.7%
- Yoruba (150): 12.0% / 22.7% vs 67.3% / 29.3%
- Hausa (150): 18.0% / 26.7% vs 68.0% / 30.0%

Mistral Large 4 (low reasoning effort; first 150 test and first 150 held-out tasks), accuracy / unsafe:

- Test: 95.3% / 2.7%; $11.28 per 1,000 checks at list ($5.64 at launch-sale); 14.4 s per check.
- Held-out: 92.0% / 2.7%; $11.72 ($5.86); 16.1 s.

Reading, not yet significance-tested:

- **Most of the gain comes from the warm start with tools.** Accuracy rises from 13-19% to 66-68% and evidence-gathering from ~0% to ~98%; following held-out attacks drops from 21% to ~12-15%.
- **RL adds a few points.** After 90 steps the naive and hardened rewards are indistinguishable, and **the hardened reward did not reduce unsafe actions more than the naive one**: both sit at 24-29%.
- **Run 2 keeps 67-73% accuracy across the five languages** after English-only training.
- **The unsafe rate stays far above Mistral's.**

Still to do: significance testing, applying the exclusions, matched 150-task comparisons with Mistral, and AuditKIT.

## Nov 7: targeted DPO repair of run 2 (rejected by the pre-agreed gate)

Design (agreed before the run; `kataa/train/dpo.py`): run 2 greedy on the 1,200 **train** tasks;
each unsafe decision becomes a pair cut at the decision: the prompt is run 2's own conversation up
to its first decision, rejected is that turn, and chosen is the scripted ideal decision turn. The
pairs go through AgentTune's self-heal path (`TrainingExample` -> `agenttune.agentic.build_dataset`
-> `create_agentic_trainer("dpo")`). Loss: sigmoid DPO, beta 0.1, plus a 0.2 SFT term on the
chosen turns. New LoRA (r 16) on merged WS2+run2, lr 2e-5, 2 epochs, effective batch 8 (72 steps).

| Job | What | Result | Cost |
|---|---|---|---|
| 6aef03f2… | Collect + pairs + DPO | Collection fine (run 2 on train: 72.5% accuracy, 23.7% unsafe, **284 pairs**). Out of memory in the first backward pass: TRL 1.7.1 takes full-length logits (~4.5K tokens x 262K vocab) before masking to the completion | ~$0.56 |
| 6aef271a… | Same pairs (reused collection), DPO with logits computed only from the first completion position (CPU-tested equal to TRL's loss), held-out eval, gate | Trained (peak 10.4 GiB, 24 s/step); **gate rejected** | ~$1.24 |

Pairs: chosen escalate 159 / no action 111 / margin call 14; rejected margin call 193 / no action
91; conditions stale_both 84, cure_active 77, oracle_conflict 75, stale_primary 27, clean 18.

Held-out (300), greedy, paired against run 2's published rows:

| | Run 2 | DPO | Difference [95% CI] | Gate check |
|---|---|---|---|---|
| Unsafe | 25.0% | 15.0% | −10.0 pp [−16.7, −3.3] | pass (CI below zero) |
| Accuracy | 73.0% | 18.3% | −54.7 pp [−62.3, −46.7] | **fail** (limit −2 pp) |
| Over-escalation | 0.3% | 81.0% | +80.7 pp [+76.0, +85.0] | **fail** (limit 5%) |
| Followed an attack | 12.3% | 15.0% | +2.7 pp [−3.3, +8.7] | **fail** (must not rise) |

**The DPO model escalates 99% of held-out tasks.** Its lower unsafe rate comes entirely from no
longer acting; the unsafe cases left (15%) are escalations that an attack asked for when the
correct answer was to act (`inj_structured` 23, `pressure_authority` 22). This is the
escalate-everything failure the gate was written to catch. Run 2 stays the final model.

Why: every pair said "this action was wrong"; none said "acting was right here". 56% of the
chosen turns are escalations and 68% of the rejected turns are margin calls, so "escalate rather
than act" fits nearly every pair. The preference accuracy hit 1.0 by step 9 and the reward margin
saturated (~5) within the first epoch (36 steps). The 0.2 SFT term on the chosen turns, mostly
escalations, did not prevent it. A fix would need anchor pairs where the correct action is chosen
over an unnecessary escalation, and fewer steps or a larger beta; that was not run.

Reproducibility: run 2 re-evaluated in the same job matches the published held-out rows on
296/300 decisions, with identical accuracy (73.0%) and unsafe rate (25.0%).
