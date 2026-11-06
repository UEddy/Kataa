"""Arithmetic check of an evaluated model (runs in: cloud session).
    python scripts/arith_check.py eval/<tag>/<sha> <model-name-prefix>

From the trajectories, against ground truth:
- ltv_tool: share of decidable cases (LTV defined) where the model computed an LTV with the calculator
- ltv_ok: share of those where that LTV is within 0.5 percentage points of the true LTV
- ages_ok: share of price quotes whose hours_between result matches the true age (within 0.1 h)
- accuracy by condition
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kataa.env.world import TaskSpec, hours_between  # noqa: E402

REPO = "JustEddy1/kataa-runs"


def tool_results(conv):
    """(name, args, result) for every executed call, pairing calls with folded results."""
    out = []
    for i, m in enumerate(conv):
        if m["role"] != "assistant" or not m.get("tool_calls"):
            continue
        nxt = conv[i + 1]["content"] if i + 1 < len(conv) else ""
        results = re.findall(r"Tool result for (\w+): (.*?)\nUse this result", nxt, re.S)
        for c, (name, res) in zip(m["tool_calls"], results):
            fn = c.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                res = json.loads(res)
            except json.JSONDecodeError:
                res = {}
            out.append((fn.get("name"), args, res))
    return out


def main():
    path, prefix = sys.argv[1], sys.argv[2]
    files = [f for f in HfApi().list_repo_files(REPO) if f.startswith(path + "/" + prefix)]
    traj = next(f for f in files if f.endswith(".trajectories.jsonl"))
    rows_f = traj.replace(".trajectories.jsonl", ".rows.jsonl")
    specs = {}
    for split in ("val", "test", "heldout", "train"):
        for line in open(Path(__file__).resolve().parents[1] / "data" / "tasks" / f"{split}.jsonl"):
            r = json.loads(line)
            specs[r["task_id"]] = TaskSpec.from_dict(json.loads(r["spec"]))
    trajs = [json.loads(x) for x in open(hf_hub_download(REPO, traj))]
    rows = [json.loads(x) for x in open(hf_hub_download(REPO, rows_f))]
    ltv_cases = ltv_tool = ltv_ok = ages_n = ages_ok = 0
    by_cond = defaultdict(lambda: [0, 0])
    for t, r in zip(trajs, rows):
        s = specs[r["task_id"]]
        by_cond[r["condition"]][0] += 1
        by_cond[r["condition"]][1] += int(r["correct"])
        calls = tool_results(t["metadata"]["conversation"])
        true_ages = {q.timestamp: hours_between(s.as_of, q.timestamp)
                     for qs in s.quotes.values() for q in qs.values()}
        for name, args, res in calls:
            if name == "hours_between" and args.get("start") in true_ages and "hours" in res:
                ages_n += 1
                ages_ok += abs(res["hours"] - true_ages[args["start"]]) <= 0.1
        if r.get("decision") and s.position_available:
            from kataa.env.world import decide
            _, ltv, _ = decide(s)
            if ltv is not None:
                ltv_cases += 1
                ltvs = [res.get("result") for n, a, res in calls
                        if n == "calculator" and "/ (" in str(a.get("expression", "")) and "result" in res]
                if ltvs:
                    ltv_tool += 1
                    ltv_ok += abs(ltvs[-1] - ltv) <= 0.5
    print(f"{traj}")
    print(f"ltv_tool {ltv_tool}/{ltv_cases} = {ltv_tool / max(ltv_cases, 1):.1%} | "
          f"ltv_ok {ltv_ok}/{ltv_tool} = {ltv_ok / max(ltv_tool, 1):.1%} | "
          f"ages_ok {ages_ok}/{ages_n} = {ages_ok / max(ages_n, 1):.1%}")
    for c, (n, k) in sorted(by_cond.items()):
        print(f"  {c:18s} n={n:3d} acc={k / n:.0%}")


if __name__ == "__main__":
    main()
