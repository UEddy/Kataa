"""Is a finished GRPO run clean and sane? Exit 0 if yes. Runs in: cloud session.
    python scripts/check_run.py <job_id> <hub run path, e.g. run1/<sha>> <expected steps>

Rule fixed before run 1 (v2) launched: the job COMPLETED with "grpo exit: 0"; all expected steps
are logged; no NaN or inf in reward, loss, grad_norm or kl; grad_norm > 1e-6 on at least 90% of
steps (it learned); max kl < 1.0 (no blow-up); mean no_decision over the last 20 steps < 0.5
(protocol not collapsed). Reward gaming does NOT fail the check: that is the experiment.
"""

import json
import math
import sys

from huggingface_hub import HfApi, hf_hub_download

REPO = "JustEddy1/kataa-runs"


def main() -> int:
    job, path, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
    fails = []
    stage = HfApi().inspect_job(job_id=job, namespace="JustEddy1").status.stage
    if stage != "COMPLETED":
        fails.append(f"job stage {stage}")
    try:
        log = open(hf_hub_download(REPO, f"{path}/logs/grpo.log", force_download=True)).read()
        if "grpo exit: 0" not in log:
            fails.append("grpo did not exit 0")
        hist = json.load(open(hf_hub_download(REPO, f"{path}/log_history.json", force_download=True)))
    except Exception as e:
        print(f"FAIL: cannot read run artifacts: {type(e).__name__}: {e}")
        return 1
    rows = [h for h in hist if "reward" in h]
    if len(rows) != steps:
        fails.append(f"{len(rows)} steps logged, expected {steps}")
    bad = [r["step"] for r in rows for k in ("reward", "loss", "grad_norm", "kl")
           if not math.isfinite(float(r.get(k, 0)))]
    if bad:
        fails.append(f"non-finite values at steps {sorted(set(bad))[:10]}")
    g = [float(r.get("grad_norm", 0)) for r in rows]
    live = sum(x > 1e-6 for x in g) / max(len(g), 1)
    if live < 0.9:
        fails.append(f"grad_norm > 1e-6 on only {live:.0%} of steps")
    kl = max((float(r.get("kl", 0)) for r in rows), default=0)
    if kl >= 1.0:
        fails.append(f"max kl {kl:.3f} >= 1.0")
    nd = [float(r.get("kataa/no_decision", 0)) for r in rows[-20:]]
    nd_mean = sum(nd) / max(len(nd), 1)
    if nd_mean >= 0.5:
        fails.append(f"no_decision over last 20 steps {nd_mean:.0%}")
    print(f"steps={len(rows)} live_grad={live:.0%} max_kl={kl:.4f} no_decision_last20={nd_mean:.0%}")
    if fails:
        print("FAIL: " + "; ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
