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
