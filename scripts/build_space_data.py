"""Build the replay data bundled into the Space. Runs in: cloud session ($0, no model calls).

    python scripts/build_space_data.py [--local DIR]     # -> space/data/

Everything comes from recorded greedy runs of the final evaluation (eval/final/88e8bac4dc62):
base Tiny Aya vs Kataa (run 2), all 300 held-out tasks, plus the five translated sets. Verdicts
are the headline scorer's rows (kataa.eval.runner.score); headline numbers come from
docs/results.json. The Space only reads these files (no Hub fetches at runtime).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kataa.rewards.episode import parse_calls_from_text  # noqa: E402
from kataa.train.common import load_split  # noqa: E402

FINAL = "eval/final/88e8bac4dc62"
HUB_FILE = "https://huggingface.co/JustEddy1/kataa-runs/blob/main/" + FINAL + "/{}"
OUT = ROOT / "space" / "data"
LANGS = ["en", "fr", "hi", "sw", "yo", "ha"]
MODELS = {"base": "base", "kataa": "run2"}
VERDICT_KEYS = ("decision", "correct", "unsafe", "followed_attack", "over_escalate", "evidence_ok",
                "n_calls")
RESULT_MAX = 700
NOTES_MAX = 1600
AGREEMENT_REF = "[the agreement text, shown in the Contract panel]"
TOOL_RESULT_RE = re.compile(r"Tool result for (\w+): ")


def split_name(lang: str) -> str:
    return "heldout" if lang == "en" else f"heldout_{lang}"


def load(base: Path, model: str, lang: str):
    stem = f"{model}__{split_name(lang)}_n0_s1_t0"
    rows = {json.loads(x)["task_id"]: json.loads(x) for x in (base / FINAL / f"{stem}.rows.jsonl").open()}
    trajs = {json.loads(x)["metadata"]["task_id"]: json.loads(x)
             for x in (base / FINAL / f"{stem}.trajectories.jsonl").open()}
    return rows, trajs, stem


def notes_of(text: str) -> str:
    """The working notes: the text before the call list."""
    for marker in ('[{"tool_name"', "[\n  {", "[{'tool_name'", "<|START_ACTION|>"):
        i = text.find(marker)
        if i >= 0:
            return text[:i].strip()
    return text.strip()


def short(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def tool_results(user_text: str) -> list[dict]:
    parts = TOOL_RESULT_RE.split(user_text)
    out = []
    for i in range(1, len(parts) - 1, 2):
        name, body = parts[i], parts[i + 1]
        body = body.replace("Use this result to answer the original question.", "").strip()
        if name == "get_loan_agreement":
            body = AGREEMENT_REF
        out.append({"name": name, "text": short(body, RESULT_MAX)})
    return out


def lane(conv: list[dict]) -> list[dict]:
    """Assistant turns with their calls and the tool results that came back."""
    turns = []
    for i, m in enumerate(conv):
        if m["role"] != "assistant":
            continue
        calls, ok = parse_calls_from_text(m["content"] or "")
        nxt = conv[i + 1] if i + 1 < len(conv) and conv[i + 1]["role"] == "user" else None
        turns.append({
            "notes": short(notes_of(m["content"] or ""), NOTES_MAX),
            "calls": [{"name": n, "args": a if isinstance(a, dict) else str(a)} for n, a, _ in (calls or [])],
            "malformed": calls is None and "[" in (m["content"] or ""),
            "results": tool_results(nxt["content"]) if nxt and nxt["content"].startswith("Tool result") else [],
        })
    return turns


def injection(spec: dict, request: str) -> dict:
    fam = spec["attack_family"]
    if fam == "inj_structured":
        return {"channel": "get_position → position_extra (structured data)",
                "text": json.dumps(spec["position_extra"], indent=1) if "workflow" in (spec["position_extra"] or {})
                else spec["position_extra"]["review_table"]}
    if fam == "pressure_authority":
        paras = [p for p in request.split("\n\n") if spec["loan_id"] not in p]
        return {"channel": "request text (authority persona)", "text": "\n\n".join(paras)}
    return {"channel": None, "text": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default=str(ROOT / "results" / "hub"))
    args = ap.parse_args()
    base = Path(args.local)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "tasks").mkdir(parents=True)

    excl = json.loads((ROOT / "data" / "translations" / "exclusions.json").read_text())
    records = {lang: {r["task_id"]: r for r in load_split(split_name(lang))} for lang in LANGS}
    data = {(lang, k): load(base, m, lang) for lang in LANGS for k, m in MODELS.items()}

    index = []
    for tid, rec in records["en"].items():
        spec = json.loads(rec["spec"])
        task = {"id": tid, "loan_id": spec["loan_id"], "as_of": spec["as_of"], "answer": rec["answer"],
                "reason": rec["reason"], "condition": rec["condition"], "tags": rec["tags"],
                "family": rec["attack_family"], "target": rec["attack_target"],
                "aligned": rec["attack_aligned"], "terms": spec["terms"],
                "agreement": spec["agreement_text"], "request": {}, "injection": {}, "lanes": {},
                "sources": {}, "excluded": {}}
        entry = {"id": tid, "condition": rec["condition"], "family": rec["attack_family"],
                 "target": rec["attack_target"], "answer": rec["answer"], "tags": rec["tags"],
                 "langs": {}}
        for lang in LANGS:
            if tid not in records[lang]:
                continue
            lrec = records[lang][tid]
            request = lrec["prompt"][-1]["content"]
            task["request"][lang] = request
            task["injection"][lang] = injection(json.loads(lrec["spec"]), request)
            task["excluded"][lang] = lang != "en" and tid in set(excl[lang]["excluded_ids"])
            task["lanes"][lang], task["sources"][lang], entry["langs"][lang] = {}, {}, {}
            for k in MODELS:
                rows, trajs, stem = data[(lang, k)]
                row, tr = rows[tid], trajs[tid]
                v = {key: row[key] for key in VERDICT_KEYS}
                task["lanes"][lang][k] = {"turns": lane(tr["metadata"]["conversation"]), "verdict": v}
                task["sources"][lang][k] = HUB_FILE.format(f"{stem}.trajectories.jsonl")
                entry["langs"][lang][k] = {"decision": row["decision"], "correct": row["correct"],
                                           "unsafe": row["unsafe"]}
            entry["langs"][lang]["excluded"] = task["excluded"][lang]
        (OUT / "tasks" / f"{tid}.json").write_text(json.dumps(task, ensure_ascii=False, separators=(",", ":")))
        index.append(entry)
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")))

    res = json.loads((ROOT / "docs" / "results.json").read_text())
    h, m = res["models"]["heldout"], res["mistral"]["heldout"]
    pick = lambda s: {k: s[k] for k in ("mean", "lo", "hi", "n")}  # noqa: E731
    summary = {
        "source": f"kataa-runs/{FINAL} (greedy, held-out split); docs/results.md",
        "heldout": {k: {mt: pick(h[mm][mt]) for mt in ("accuracy", "unsafe", "unsafe_when_deciding",
                                                      "followed_attack", "over_escalate", "no_decision")}
                    for k, mm in MODELS.items()},
        "mistral_matched": {k: {mt: pick(m[mm][mt]) for mt in ("accuracy", "unsafe", "over_escalate")}
                            for k, mm in (("kataa", "run2"), ("mistral", "mistral"))},
        "mistral_cost_per_1000_list": m["cost"]["usd_per_1000_checks_list_2026_10_22"],
        "kataa_cost_per_1000": 0.32,
        "dpo": {"unsafe": [0.25, 0.15], "accuracy": [0.73, 0.183], "over_escalate": [0.003, 0.81],
                "escalated": 0.99},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    size = sum(p.stat().st_size for p in OUT.rglob("*.json"))
    print(f"[space] {len(index)} tasks, {size / 1e6:.1f} MB in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
