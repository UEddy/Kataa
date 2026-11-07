"""Assemble (and optionally upload) the model repo JustEddy1/kataa. Runs in: cloud session ($0).

    python scripts/build_model_repo.py [--upload]     # -> results/kataa-model/

Contents: warmstart/ (WS2 adapter, base CohereLabs/tiny-aya-global), run2/ (run 2's GRPO adapter,
trained on the MERGED warm start), load_kataa.py, and the model card (docs/model-card.md).

run2's adapter_config.json named its base as /work/start-merged (a path inside the training job).
It is rewritten to a name that cannot be loaded on its own, so `AutoPeftModel` on run2/ fails
loudly instead of silently dropping the warm start (the bug behind eval/run1/8c050d7f4788).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "kataa-model"
REPO = "JustEddy1/kataa"
RUNS = "JustEddy1/kataa-runs"
SOURCES = {"warmstart": "warmstart/f7118070bac9", "run2": "run2/169140127fca"}
RUN2_BASE = "JustEddy1/kataa/warmstart merged into CohereLabs/tiny-aya-global (see README)"

LOADER = '''"""Load Kataa: Tiny Aya Global + the warm-start adapter + run 2's adapter, merged in order."""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE, REPO = "CohereLabs/tiny-aya-global", "JustEddy1/kataa"


def load_kataa(dtype=torch.bfloat16, device_map=None):
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype, device_map=device_map)
    model = PeftModel.from_pretrained(model, REPO, subfolder="warmstart").merge_and_unload()
    model = PeftModel.from_pretrained(model, REPO, subfolder="run2").merge_and_unload()
    return tok, model


if __name__ == "__main__":
    tok, model = load_kataa()
    print(type(model).__name__, sum(p.numel() for p in model.parameters()) / 1e9, "B parameters")
'''


def build() -> None:
    from huggingface_hub import snapshot_download

    if OUT.exists():
        shutil.rmtree(OUT)
    src = ROOT / "results" / "hub-adapters"
    snapshot_download(RUNS, local_dir=src,
                      allow_patterns=[f"{p}/adapter_*" for p in SOURCES.values()])
    for name, p in SOURCES.items():
        (OUT / name).mkdir(parents=True)
        for f in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copy(src / p / f, OUT / name / f)
        cfg = json.loads((OUT / name / "adapter_config.json").read_text())
        cfg["base_model_name_or_path"] = ("CohereLabs/tiny-aya-global" if name == "warmstart"
                                          else RUN2_BASE)
        (OUT / name / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"[model] {name} <- {RUNS}/{p}: base={cfg['base_model_name_or_path']!r} r={cfg['r']}")
    (OUT / "load_kataa.py").write_text(LOADER)
    shutil.copy(ROOT / "docs" / "model-card.md", OUT / "README.md")


def upload() -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    assert api.repo_info(REPO).private, "expected the private repo"
    api.upload_folder(folder_path=str(OUT), repo_id=REPO,
                      commit_message="Kataa: warm-start and run 2 adapters, loader, model card")
    print(f"[model] uploaded to {REPO}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    build()
    if args.upload:
        upload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
