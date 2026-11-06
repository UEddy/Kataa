"""Summarise a GRPO run's training log in windows (reward-hacking evidence). Runs in: cloud session.
    python scripts/analyze_run.py run1/<sha> [--window 20]"""

import argparse
import json

from huggingface_hub import hf_hub_download

KEYS = ["reward", "kataa/correct", "kataa/evidence_ok", "kataa/unsafe", "kataa/over_escalate",
        "kataa/followed_attack", "kataa/no_decision", "kataa/decided_margin_call",
        "kataa/decided_no_action", "kataa/decided_escalate", "kataa/calls", "kataa/format_ok",
        "kl", "step_time"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--repo", default="JustEddy1/kataa-runs")
    a = ap.parse_args()
    hist = json.load(open(hf_hub_download(a.repo, f"{a.path}/log_history.json")))
    rows = [h for h in hist if "reward" in h]
    print(f"{len(rows)} steps")
    head = "steps".ljust(10) + "".join(k.split("/")[-1][:11].rjust(12) for k in KEYS)
    print(head)
    for i in range(0, len(rows), a.window):
        w = rows[i:i + a.window]
        vals = [sum(float(r.get(k, 0)) for r in w) / len(w) for k in KEYS]
        print(f"{w[0]['step']}-{w[-1]['step']}".ljust(10) + "".join(f"{v:12.4f}" for v in vals))


if __name__ == "__main__":
    main()
