"""Translate the held-out split's requests and injected text. Runs in: cloud session.

    python scripts/translate_heldout.py [--langs fr,hi,sw,yo,ha] [--limit N]

Translator: mistral-large-4 (not Tiny Aya, the model under test). Mistral is not evaluated on
the translated sets. Only the user request (including pressure text) and the injected strings in
structured-injection attacks are translated; the system prompt, tool outputs and the agreement
stay in English, as in deployment. Every item is back-translated to English and checked:
protected tokens (loan IDs, numbers, snake_case tool names) must survive, and word overlap with
the original is recorded (low overlap is flagged). Results are cached per language, so the
script resumes where it stopped. Writes data/tasks_translated/heldout_<lang>.jsonl and
data/translations/report.json.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kataa.eval.mistral_client import MistralClient  # noqa: E402

LANGS = {"fr": "French", "hi": "Hindi", "sw": "Swahili", "yo": "Yoruba", "ha": "Hausa"}
BATCH = 6
PROTECT = re.compile(r"LN-[0-9A-F]{6}|\d{2} [A-Z][a-z]{2} \d{4}|\b[a-z]+_[a-z_]+\b|\d+(?:[.:]\d+)*")
INSTRUCT = (
    "Translate each string in the JSON array below into {lang}. Keep these exactly as written: "
    "loan IDs such as LN-1A2B3C, all numbers, dates and times (use Western digits), tool names "
    "written in snake_case, and markdown table structure (| and ---). Translate everything "
    "else naturally, including instructions, names of offices and titles. Reply with only a "
    "JSON array of the translated strings, with the same length and order.\n\n{payload}"
)


def units(rec: dict) -> list[str]:
    """Texts to translate for one task: the request, then injected strings (if any)."""
    spec = json.loads(rec["spec"])
    out = [spec["request_text"]]
    extra = spec.get("position_extra") or {}
    if "review_table" in extra:
        out.append(extra["review_table"])
    d = (extra.get("workflow") or {}).get("automation_directive")
    if d:
        out.append(d["rationale"])
    return out


def apply(rec: dict, tr: list[str], lang: str) -> dict:
    spec = json.loads(rec["spec"])
    spec["request_text"] = tr[0]
    extra = copy.deepcopy(spec.get("position_extra") or {})
    i = 1
    if "review_table" in extra:
        extra["review_table"] = tr[i]
        i += 1
    if (extra.get("workflow") or {}).get("automation_directive"):
        extra["workflow"]["automation_directive"]["rationale"] = tr[i]
    spec["position_extra"] = extra or None
    out = dict(rec)
    out["prompt"] = [rec["prompt"][0], {"role": "user", "content": tr[0]}]
    out["spec"] = json.dumps(spec, sort_keys=True)
    out["split"] = f"heldout_{lang}"
    out["lang"] = lang
    return out


def key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def overlap(a: str, b: str) -> float:
    wa, wb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(wa & wb) / max(len(wa | wb), 1)


def call_batch(client: MistralClient, texts: list[str], lang: str) -> list[str]:
    for attempt in range(3):
        reply, _ = client.chat([{"role": "user", "content": INSTRUCT.format(
            lang=lang, payload=json.dumps(texts, ensure_ascii=False))}])
        m = re.search(r"\[.*\]", reply, re.S)
        try:
            arr = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            arr = None
        if isinstance(arr, list) and len(arr) == len(texts) and all(isinstance(x, str) for x in arr):
            return arr
    raise RuntimeError(f"bad translation reply for {lang}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    recs = [json.loads(x) for x in (ROOT / "data/tasks/heldout.jsonl").open()][: args.limit]
    client = MistralClient(max_tokens=4096)
    report = {}
    for code in args.langs.split(","):
        lang = LANGS[code]
        cache_path = ROOT / f"data/translations/cache_{code}.jsonl"
        cache = {}
        if cache_path.exists():
            for line in cache_path.open():
                c = json.loads(line)
                cache[c["key"]] = c
        todo = [t for t in dict.fromkeys(u for r in recs for u in units(r)) if key(t) not in cache]
        print(f"[{code}] {len(todo)} texts to translate ({len(cache)} cached)", flush=True)
        with cache_path.open("a") as f:
            for i in range(0, len(todo), BATCH):
                src = todo[i:i + BATCH]
                fwd = call_batch(client, src, lang)
                back = call_batch(client, fwd, "English")
                for s, t, b in zip(src, fwd, back):
                    lost = sorted(set(PROTECT.findall(s)) - set(PROTECT.findall(t)))
                    c = {"key": key(s), "source": s, "translation": t, "back": b,
                         "protected_lost": lost, "back_overlap": round(overlap(s, b), 3)}
                    cache[c["key"]] = c
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{code}] {min(i + BATCH, len(todo))}/{len(todo)}", flush=True)
        out = [apply(r, [cache[key(u)]["translation"] for u in units(r)], code) for r in recs]
        with (ROOT / f"data/tasks_translated/heldout_{code}.jsonl").open("w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in out)
        items = [cache[key(u)] for r in recs for u in units(r)]
        ov = sorted(c["back_overlap"] for c in items)
        report[code] = {"items": len(items), "protected_lost": sum(bool(c["protected_lost"]) for c in items),
                        "back_overlap_median": ov[len(ov) // 2], "back_overlap_p10": ov[len(ov) // 10],
                        "flagged_low_overlap": sum(o < 0.35 for o in ov)}
        print(f"[{code}] {report[code]}", flush=True)
    usage = client.usage_log
    report["_usage"] = {"requests": len(usage), "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
                        "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage)}
    rp = ROOT / "data/translations/report.json"
    old = json.loads(rp.read_text()) if rp.exists() else {}
    old.update(report)
    rp.write_text(json.dumps(old, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
