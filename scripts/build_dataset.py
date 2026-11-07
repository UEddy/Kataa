"""Build (and optionally upload) the Kataa benchmark dataset. Runs in: cloud session ($0).

    python scripts/build_dataset.py [--upload]      # -> results/kataa-bench/

Layout (one `datasets` config per folder, declared in the card's YAML):
  tasks/{train,val,test,heldout}.jsonl          the benchmark tasks (config "tasks")
  translated/heldout_{fr,hi,sw,yo,ha}.jsonl     held-out requests and injected text, translated,
                                                with excluded_from_metrics (config "translated")
  ideal_runs/{train,val}.jsonl                  scripted ideal conversations, the warm-start data
                                                (config "ideal_runs")
  translation_audit/{cache_*.jsonl, report.json, exclusions.json}   every translated text with
                                                its back-translation, protected-token check and
                                                overlap score (config "translation_audit")
  prices/coinmetrics_daily.csv                  the real price snapshot the tasks replay
  README.md                                     the dataset card (docs/dataset-card.md)

--upload pushes to datasets/JustEddy1/kataa-bench (private until Eddy makes it public).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "kataa-bench"
REPO = "JustEddy1/kataa-bench"
LANGS = ["fr", "hi", "sw", "yo", "ha"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build() -> dict:
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in ("tasks", "translated", "ideal_runs", "translation_audit", "prices"):
        (OUT / d).mkdir(parents=True)
    for split in ("train", "val", "test", "heldout"):
        shutil.copy(ROOT / "data" / "tasks" / f"{split}.jsonl", OUT / "tasks" / f"{split}.jsonl")
    excl = json.loads((ROOT / "data" / "translations" / "exclusions.json").read_text())
    counts = {}
    for lang in LANGS:
        drop = set(excl[lang]["excluded_ids"])
        rows = [json.loads(x) for x in (ROOT / "data" / "tasks_translated" / f"heldout_{lang}.jsonl").open()]
        with (OUT / "translated" / f"heldout_{lang}.jsonl").open("w") as f:
            for r in rows:
                r["excluded_from_metrics"] = r["task_id"] in drop
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[lang] = {"tasks": len(rows), "excluded": sum(r["task_id"] in drop for r in rows)}
        assert counts[lang]["excluded"] == len(drop), lang
    sft = ROOT / "data" / "sft"
    if not (sft / "train.jsonl").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_sft.py")], check=True)
    for split in ("train", "val"):
        shutil.copy(sft / f"{split}.jsonl", OUT / "ideal_runs" / f"{split}.jsonl")
    for p in sorted((ROOT / "data" / "translations").iterdir()):
        shutil.copy(p, OUT / "translation_audit" / p.name)
    prices = ROOT / "data" / "prices" / "coinmetrics_daily.csv"
    shutil.copy(prices, OUT / "prices" / prices.name)
    assert sha256(prices) == "4e0258efb91a2444cf36916ee0504f0df6e45cf92c2c976b67d1b339b180abc5"
    shutil.copy(ROOT / "docs" / "dataset-card.md", OUT / "README.md")
    return counts


def validate() -> None:
    """Every config in the card loads with `datasets`, from the local folder, before upload."""
    import yaml
    from datasets import load_dataset

    card = (OUT / "README.md").read_text()
    meta = yaml.safe_load(card.split("---")[1])
    for cfg in meta["configs"]:
        files = {d["split"]: str(OUT / d["path"]) for d in cfg["data_files"]}
        ds = load_dataset("json", data_files=files)
        print(f"[dataset] {cfg['config_name']}: " + ", ".join(f"{k}={len(v)}" for k, v in ds.items()))
        for split, d in ds.items():
            assert len(d) > 0, (cfg["config_name"], split)


def upload() -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    assert api.repo_info(REPO, repo_type="dataset").private, "expected the private repo"
    api.upload_folder(folder_path=str(OUT), repo_id=REPO, repo_type="dataset",
                      commit_message="Kataa benchmark: tasks, translations, ideal runs, audit, prices")
    print(f"[dataset] uploaded to datasets/{REPO}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    counts = build()
    print(f"[dataset] translated: {json.dumps(counts)}")
    validate()
    if args.upload:
        upload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
