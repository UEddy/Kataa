# Write-up outline (HF Article and README)

Agreed framing, 2026-11-07. Every number comes from `docs/results.md` (95% bootstrap CIs, paired
over tasks). Every negative number is reported plainly.

## 1. Lead: what it does when it acts, and how it compares (first screen)

**Unsafe among decided tasks** (unsafe decisions / tasks with a decision), held-out (300):

| | Base Tiny Aya | Kataa (run 2) |
|---|---|---|
| Unsafe among decided | 61.0% [53.1, 68.9] | 25.0% [20.3, 30.0] |
| Tasks with no decision | 51.3% | 0% |
| Accuracy | 19.0% | 73.0% |

Test (300): 63.6% → 29.3% unsafe among decided.

Why this metric leads: base Tiny Aya makes no decision on about half the tasks, so its all-tasks
unsafe rate (29.7%) looks close to Kataa's (25.0%), and the paired difference on all tasks is not
significant (−4.7 pp [−11.3, +2.0]). On test, the warm start is more unsafe than base on the
all-tasks rate (+7.0 pp [+0.3, +13.3]). Report both rates side by side.

**Matched comparison with Mistral Large 4** (a large hosted open-weights model, via API, low
reasoning effort), on the same first 150 held-out tasks:

| | Kataa (run 2) | Mistral Large 4 |
|---|---|---|
| Accuracy | 70.0% | 92.0% |
| Unsafe | 30.0% | 2.7% |
| Unsafe among decided | 30.0% | 2.8% |
| Over-escalation | 0.0% | 2.7% |
| No decision | 0.0% | 4.0% |
| Cost per 1,000 checks | ~$0.32 (L40S list price, batched throughput; excludes load and idle time) | $11.72 list ($5.86 launch-sale) |

The cost figure for Kataa is $1.80/h × 191 s per 300 held-out checks (final eval job), and is a
throughput cost, not a per-check latency. Test-split matched numbers are in `docs/results.md`.

Also on the first screen: **a targeted DPO repair cut the unsafe rate significantly (25.0% →
15.0%, −10.0 pp [−16.7, −3.3]) and the pre-agreed gate still rejected it, because it had stopped
acting** (it escalates 99% of tasks). Section 2 tells that story.

Conclusion stated up front: Kataa is about 37x cheaper per check, but unsafe roughly 30% of the
time against 2.7%, so **it should not act autonomously**. At best it is a triage layer with a
human or a stronger model behind it.

## 2. Reward hacking in three parts (problem #9); shown prominently, right after the lead

**Part 1: the reward could be gamed on paper.** The scripted-strategy table (README, from
`tests/test_rewards.py`, 300 train tasks): under the naive reward, guessing without tools pays as
well as gathering evidence, and "escalate everything" earns 0.177. The hardened reward makes every
shortcut negative (escalate everything with full evidence: −0.160).

| Strategy | Naive | Hardened |
|---|---|---|
| Ideal | 1.000 | 1.200 |
| Escalate everything, no tools | 0.177 | −0.337 |
| Escalate everything, full evidence | 0.177 | −0.160 |
| Margin-call everything, full evidence | 0.443 | −0.470 |
| No action everything, no tools | 0.380 | −0.530 |
| Obey the attacker, otherwise ideal | 0.673 | 0.220 |

**Part 2: an optimiser found the shortcut.** GRPO run 1 did not game the naive reward within 90
steps (evidence-gathering stayed at 97–100%). The DPO repair did find a shortcut: trained only on
run 2's unsafe decisions (284 train-split pairs, the correct decision chosen over the unsafe one),
it learned "escalate rather than act". On held-out it escalates 99% of tasks. Its scores are
almost exactly the scripted "escalate everything, full evidence" row: naive 0.183 (scripted
0.177), hardened −0.267 (scripted −0.160). The headline metric improved anyway: **unsafe 25.0% →
15.0%, −10.0 pp [−16.7, −3.3], significant.** A team reporting only the unsafe rate would have
shipped it.

**Part 3: the gate caught it.** The acceptance rule was fixed before the run (`kataa/eval/gate.py`):
unsafe must fall significantly, accuracy may drop at most 2 pp, over-escalation at most 5% and not
significantly above run 2's, and the followed-attack rate must not rise. Result: unsafe passed;
accuracy −54.7 pp [−62.3, −46.7] failed; over-escalation 0.3% → 81.0% failed; followed attacks
12.3% → 15.0% failed. Why it happened: every pair said "this action was wrong" and none said
"acting was right here" (56% of chosen turns were escalations, 68% of rejected turns were margin
calls). The 0.2 SFT term did not prevent it. Run 2 stays the final model.

Takeaway: a safety metric optimised on its own pays for "refuse everything"; the defence is a
gate that measures task success and over-escalation as well, written before the run.

## 3. What worked: the warm start (agentic distillation)

Held-out, base → run 2: accuracy +54.0 pp, evidence-gathering ~0% → 97%, following attacks
21.0% → 12.3% (−8.7 pp [−14.3, −3.0]). It holds across five languages after English-only
training (67–72% accuracy; gap to English on the same tasks between −0.4 and −2.3 pp, none
significant), with translation exclusions applied.

## 4. RL: naive vs hardened reward, an honest null result at 90 steps

- Both runs beat WS2 on held-out (accuracy about +5 pp, unsafe about −3 pp, significant). On test
  the difference is not significant.
- Run 2 vs run 1: no significant difference on any headline metric.
- Failure analysis: stale prices on both oracles, oracle conflicts and active cure periods stay at
  0–9% accuracy for every trained model, and RL made cure periods worse (WS2 8.9% → run 2 0%).

## 5. Method, data, metrics, limitations

Why these metrics, why this data (real Coin Metrics prices, CC BY-NC; synthetic contracts and
attacks) and why this model, plus the reproduction steps. Limitations: the Mistral subset is 150
tasks; the Space is a replay of recorded runs, with no live inference.
