"""Apply the pre-agreed DPO acceptance rule. Runs in: HF Job (end of dpo_repair.sh) or cloud session.

    python -m kataa.eval.gate --dpo dpo__heldout.rows.jsonl [--rerun run2__heldout.rows.jsonl] \
        --out gate.json [--hub-path dpo/<sha>]

The baseline is run 2's published final-eval rows (eval/final/88e8bac4dc62, greedy, held-out
300), fixed before the DPO run. `--rerun` is run 2 evaluated again in the same job: it is only
reported, as a check on vLLM greedy reproducibility, and does not change the verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kataa.eval.stats import dpo_gate, load_rows, paired_diff

BASELINE = "eval/final/88e8bac4dc62/run2__heldout_n0_s1_t0.rows.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo", required=True)
    ap.add_argument("--rerun", default=None)
    ap.add_argument("--baseline", default=None, help="local copy of the run-2 baseline rows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hub-path", default=None)
    args = ap.parse_args()

    if args.baseline:
        base_path = args.baseline
    else:
        from huggingface_hub import hf_hub_download
        from kataa.train.hub import RUNS_REPO
        base_path = hf_hub_download(RUNS_REPO, BASELINE)
    run2, dpo = load_rows(base_path), load_rows(args.dpo)
    assert len(run2) == 300 and set(dpo) == set(run2), "DPO eval must cover the same 300 tasks"
    res = {"baseline": BASELINE, "gate": dpo_gate(run2, dpo)}
    if args.rerun:
        rerun = load_rows(args.rerun)
        res["run2_rerun_vs_published"] = {
            "identical_decisions": sum(rerun[i]["decision"] == run2[i]["decision"] for i in run2),
            "n": len(run2),
            **{m: paired_diff(run2, rerun, m) for m in ("accuracy", "unsafe")}}
    out = Path(args.out)
    out.write_text(json.dumps(res, indent=1))
    g = res["gate"]
    print(f"[gate] accepted={g['accepted']} checks={json.dumps(g['checks'])}", flush=True)
    for m in ("unsafe", "accuracy", "over_escalate", "followed_attack"):
        d = g[m]
        print(f"[gate] {m}: run2 {d['a']:.3f} -> dpo {d['b']:.3f}, diff {d['diff']:+.3f} "
              f"[{d['lo']:+.3f}, {d['hi']:+.3f}]", flush=True)
    if args.hub_path:
        from kataa.train.hub import push_folder
        push_folder(out.parent, args.hub_path, f"{args.hub_path}: gate", allow=[out.name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
