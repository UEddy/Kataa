# Space plan: recorded-run replay (agreed with Eddy, 2026-11-07)

## Decision

The Space **replays recorded evaluation runs**. It never runs a model. Every episode shown is a
greedy run from the final evaluation job (`kataa-runs/eval/final/88e8bac4dc62`, job 6aee9bf3…),
and the page says so above the fold and on every episode: "Recorded run, held-out split, greedy.
No live inference."

## Hardware: a static Space, free

Checked on 2026-11-07 against <https://huggingface.co/docs/hub/spaces-overview>:

> Static Spaces are free for everyone. Gradio and Docker Spaces run on compute and require a paid
> plan to create: PRO for personal accounts, Team or Enterprise for organizations. Free personal
> accounts in good standing can still host up to 2 Gradio Spaces running on ZeroGPU.

That also explains the HTTP 402 when we tried to create a private Space earlier. So:

- **SDK: `static`** (HTML, CSS and JS served as files). It needs no hardware and costs $0, with
  no dependency on CPU Basic or ZeroGPU.
- A Gradio Space on CPU Basic (2 vCPU, 16 GB, free hardware) would need PRO on this account. Live
  inference would need a GPU Space (T4-small, $0.40/h) or ZeroGPU. Neither is needed for a replay.
- Visibility: **public**, created only when the demo is ready (Eddy's rule).

## Content

Data comes from a build script, `scripts/build_space_data.py` (cloud session, $0), that reads the
recorded `*.trajectories.jsonl` and `*.rows.jsonl` from the runs repo and writes compact JSON
into the Space:

- **All 300 held-out tasks**, base Tiny Aya next to Kataa (run 2), on the same task. If the DPO
  repair passes its gate, it is added as a third, clearly labelled lane. It does not replace run 2.
- For each episode: the request text, the condition, the attack family and channel, the injected
  text (highlighted where it sits in the tool output), the ground truth with its reason, and each
  model's turns (tool calls with arguments, shortened tool results that expand on click, working
  notes). Then the decision and reason, and the verdict: correct, unsafe, followed the attack,
  over-escalated, or no decision.
- **Language toggle** for the five translated sets (base and run 2 only; excluded task IDs are
  marked "excluded from metrics: translation check failed").
- **Filters:** attack family (`inj_structured`, `pressure_authority`), condition, outcome. The
  outcome filter includes **"Kataa unsafe"**, so Kataa's failures (stale prices, oracle conflicts,
  cure periods) are one click away. Nothing is cherry-picked: the default view is a fixed random
  sample, seed 0.
- **Headline strip:** unsafe among decided tasks (base 61.0% → Kataa 25.0% on held-out) and the
  matched Mistral numbers. Every number links to `docs/results.md`.
- Links: the dataset, the model, the Article, and the exact Hub file each episode came from.

Size budget: the recorded held-out trajectories for two models are about 2.5 MB of JSON before
trimming. Agreement text is stored once per task, not once per model. The target is under 10 MB
total, loaded lazily per task.

## Deviation from the brief

The brief says a judge can "pick **or paste** a poisoned feed". Pasting would need live
inference, so this plan supports **pick only**. The Article states that clearly. If Eddy wants
paste later, the cheapest route is a ZeroGPU Gradio Space (free tier, 2 per personal account).

## Design direction (not default Gradio)

- A margin-desk "ledger" look: an off-white paper ground, ink-black monospace for tool traffic, a
  serif for the contract text, and one signal colour (red) used only for injected text and unsafe
  verdicts. Full dark mode.
- Two lanes side by side (base | Kataa) on a shared timeline, one row per turn, so you can see
  where the two runs diverge. Verdict stamps at the end of each lane ("MARGIN CALL: UNSAFE",
  "ESCALATED: CORRECT").
- Injected instructions are shown inline, struck through with a margin note: "instruction found
  in data, not from the desk".
- Usable at phone width (lanes stack). No external fonts or scripts beyond what the static Space
  serves.
