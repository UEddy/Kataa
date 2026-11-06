# Kataa

*Kataa* is Swahili for "refuse". Kataa is a small open model (Tiny Aya Global, 3.35B) fine-tuned with AgentTune to take consequential financial actions, here margin calls on crypto-backed loans, only when they're justified. It refuses when the request is manipulated, the data is stale or contradictory, or the evidence isn't there. This is an entry for the Lexsi Labs AI Safety and Alignment Hackathon, Track 3 (Agentic Fine-tuning).

Status: work in progress. The environment, real price replay, attack families and task splits are done. Rewards, training and evaluation come next.

- Data, licences, ground-truth rules and splits: [DATA.md](DATA.md)
- Cloud-session setup: `scripts/cloud-setup.sh`
- Regenerate tasks: `python -m kataa.env.generate`
- Tests: `/opt/kataa-venv/bin/python -m pytest`

Licences: Tiny Aya is CC-BY-NC. AgentTune and AuditKIT are under Lexsi Labs' source-available non-commercial licences. Price data is © Coin Metrics, Inc., under CC BY-NC 4.0.
