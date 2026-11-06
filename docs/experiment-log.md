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
