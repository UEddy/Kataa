"""Resumable Mistral Large 4 evaluation. Runs in: cloud session (background, chunked).

    python -m kataa.eval.mistral_eval --split test --n-tasks 150 --out results/mistral \
        --hub-path eval/mistral-large-4 [--max-episodes 40]

Same protocol as Tiny Aya (system prompt with the tool schemas, the JSON call list, AgentTune's
parser and tool-result folding, temperature 0). One episode at a time; each finished episode is
appended to <out>/<split>.rows.jsonl and .trajectories.jsonl (with its token usage), already
finished task_ids are skipped, and results are pushed to the Hub every 10 episodes. Mistral is
evaluated on English test and held-out subsets only, never on the translated sets.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kataa.eval.mistral_client import PRICES, MistralClient, cost_usd
from kataa.eval.runner import run_episodes, score, summarize, to_agenttune_jsonl
from kataa.train.common import load_split
from kataa.train.hub import push_folder


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["test", "heldout", "val"])
    ap.add_argument("--n-tasks", type=int, default=150)
    ap.add_argument("--out", default="results/mistral")
    ap.add_argument("--hub-path", default="eval/mistral-large-4")
    ap.add_argument("--max-episodes", type=int, default=None, help="stop after N new episodes")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_p, traj_p = out / f"{args.split}.rows.jsonl", out / f"{args.split}.trajectories.jsonl"
    if not rows_p.exists():  # session was wiped: resume from the Hub copy
        from huggingface_hub import hf_hub_download
        from kataa.train.hub import RUNS_REPO
        for p in (rows_p, traj_p, out / "summary.json"):
            try:
                src = hf_hub_download(RUNS_REPO, f"{args.hub_path}/{p.name}", force_download=True)
                p.write_bytes(Path(src).read_bytes())
            except Exception:
                pass
    done = {json.loads(x)["task_id"] for x in rows_p.open()} if rows_p.exists() else set()
    todo = [r for r in load_split(args.split, args.n_tasks) if r["task_id"] not in done]
    print(f"[mistral] {args.split}: {len(done)} done, {len(todo)} to go", flush=True)
    client = MistralClient()
    new = 0
    for rec in todo:
        if args.max_episodes is not None and new >= args.max_episodes:
            break
        start = len(client.usage_log)
        t0 = time.time()

        def generate(convs):
            return [client.chat(c)[0] for c in convs]

        ep = run_episodes(generate, [rec])[0]
        used = client.usage_log[start:]
        usage = {"requests": len(used),
                 "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in used),
                 "cached_tokens": sum((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
                                      for u in used),
                 "completion_tokens": sum(u.get("completion_tokens", 0) for u in used),
                 "reasoning_tokens": sum((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
                                         for u in used),
                 "latency_s": round(sum(u.get("latency_s", 0) for u in used), 1),
                 "wall_s": round(time.time() - t0, 1)}
        for name, price in PRICES.items():
            usage[f"cost_usd_{name}"] = round(sum(cost_usd(u, price) for u in used), 6)
        row = {**score([ep])[0], "usage": usage}
        traj = to_agenttune_jsonl(ep)
        traj["metadata"]["usage"] = usage
        with rows_p.open("a") as f:
            f.write(json.dumps(row) + "\n")
        with traj_p.open("a") as f:
            f.write(json.dumps(traj) + "\n")
        new += 1
        print(f"[mistral] {rec['task_id']} decision={row['decision']} answer={row['answer']} "
              f"calls={row['n_calls']} tokens={usage['prompt_tokens']}+{usage['completion_tokens']} "
              f"wall={usage['wall_s']}s", flush=True)
        if new % 10 == 0:
            write_summary(out, args.split)
            push_folder(out, args.hub_path, f"{args.hub_path}: {args.split} {len(done) + new}")
    write_summary(out, args.split)
    push_folder(out, args.hub_path, f"{args.hub_path}: {args.split} {len(done) + new}")
    return 0


def write_summary(out: Path, split: str) -> None:
    rows = [json.loads(x) for x in (out / f"{split}.rows.jsonl").open()]
    s = summarize(rows)
    n = len(rows)
    for key in ("requests", "prompt_tokens", "completion_tokens", "reasoning_tokens", "latency_s"):
        s[f"{key}_per_check"] = sum(r["usage"][key] for r in rows) / n
    for name in PRICES:
        s[f"usd_per_1000_checks_{name}"] = 1000 * sum(r["usage"][f"cost_usd_{name}"] for r in rows) / n
    path = out / "summary.json"
    all_s = json.loads(path.read_text()) if path.exists() else {}
    all_s[split] = s
    path.write_text(json.dumps(all_s, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
