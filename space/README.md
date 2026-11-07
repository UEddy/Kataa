---
title: Kataa · recorded margin-desk runs
emoji: 🛑
colorFrom: red
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: cc-by-nc-4.0
short_description: Base Tiny Aya vs Kataa on held-out attacks, replayed
models:
- JustEddy1/kataa
- CohereLabs/tiny-aya-global
datasets:
- JustEddy1/kataa-bench
---

# Kataa: recorded runs

A replay of recorded evaluation runs: base Tiny Aya Global vs Kataa (Tiny Aya fine-tuned with
AgentTune), side by side on the same 300 held-out tasks, in six languages. **No model runs in
this Space**: every episode is a greedy run from the final evaluation job, bundled as JSON in
`data/`, with a link to the exact recorded file in
[JustEddy1/kataa-runs](https://huggingface.co/JustEddy1/kataa-runs).

Collection: [JustEddy1/kataa](https://huggingface.co/collections/JustEddy1/kataa) ·
Model: [JustEddy1/kataa](https://huggingface.co/JustEddy1/kataa) ·
Benchmark: [JustEddy1/kataa-bench](https://huggingface.co/datasets/JustEddy1/kataa-bench) ·
Code: [github.com/UEddy/Kataa](https://github.com/UEddy/Kataa) ·
Write-up: [README](https://github.com/UEddy/Kataa#readme) (official), [long form](https://github.com/UEddy/Kataa/blob/main/docs/article.md)

Licences: Tiny Aya and Kataa are CC-BY-NC; AgentTune and AuditKIT are Lexsi Labs' non-commercial
licences; price data © Coin Metrics, Inc., CC BY-NC 4.0.
