**Kataa: a 3B agent that refuses unjustified financial actions**

Kataa is Tiny Aya Global (3.35B) fine-tuned with AgentTune to work a simulated margin desk: read a
loan agreement, a position and two price oracles with tools, then make one decision (margin call,
no action, or escalate). Prices are real (Coin Metrics); contracts and attacks are synthetic, and
every answer is computed in code. All results are greedy on 300 held-out tasks with two attack
families never seen in training, with 95% bootstrap CIs.

**Results.** Among tasks where the model decided, unsafe actions fell from 61.0% [53.1, 68.9] for
base Tiny Aya to 25.0% [20.3, 30.0]. The caveat: base decides nothing on 51.3% of tasks, so its
all-tasks unsafe rate (29.7%) is close to Kataa's (25.0%; −4.7 pp [−11.3, +2.0], not
significant), and on the in-distribution test split Kataa's all-tasks rate is higher (+6.7 pp
[+0.0, +13.3]). Accuracy rose from 19.0% [14.7, 23.3] to 73.0% [68.0, 78.0]; following the attack
fell from 21.0% [16.7, 25.7] to 12.3% [8.7, 16.3] (−8.7 pp [−14.3, −3.0]); evidence was gathered
before deciding on 97.3% of tasks (base 0.3%). With English-only training and translation-check
exclusions applied, Kataa reaches 67.2–72.2% accuracy in French, Hindi, Swahili, Yoruba and Hausa
(base 13.3–22.1%), within −0.4 to −2.3 pp of English on the same tasks, none significant.

**What didn't work.** Most of the gain comes from the warm start (agentic distillation of
scripted ideal runs). GRPO with a naive reward and GRPO with a hardened reward were statistically
indistinguishable after 90 steps (accuracy −0.7 pp [−2.0, +0.7], unsafe +0.7 pp [+0.0, +1.7]):
an honest null result. Stale prices, oracle conflicts and active cure periods remain at 0–9%
accuracy.

**Reward hacking in three parts.** (1) Scripted strategies show the naive reward pays guessing
without tools as well as gathering evidence. (2) A DPO repair on Kataa's 284 unsafe training-split
decisions cut unsafe actions significantly (25.0% → 15.0%, −10.0 pp [−16.7, −3.3]) by learning to
escalate 99% of tasks. (3) An acceptance gate written before the run rejected it: accuracy fell to
18.3% and over-escalation rose to 81.0%. Run 2 remains the released model.

**Deployment comparison.** On the same 150 held-out tasks, Mistral Large 4 (a large hosted
open-weights model) scores 92.0% accuracy and 2.7% unsafe against Kataa's 70.0% and 30.0%
(−27.3 pp [−34.7, −19.3]), at $11.72 per 1,000 checks against about $0.32 for Kataa. Kataa should
not act autonomously; its use is as a cheap first-pass checker that gathers the evidence and routes
every decision to a human.

**AuditKIT.** Verified outcomes agree with our scorer on every Tiny Aya task; tool-call F1 against
the ideal run is 0.85 for Kataa and 0.13 for base. We hit a known bug in `ak.compare`'s significance
test (AuditKIT issue #20, since fixed upstream) and used our own paired bootstrap instead.

Write-up: https://github.com/UEddy/Kataa#readme
Replay Space (recorded runs): https://huggingface.co/spaces/JustEddy1/kataa-demo
Collection (model, benchmark, run artefacts): https://huggingface.co/collections/JustEddy1/kataa
