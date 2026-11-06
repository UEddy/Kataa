from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "CohereLabs/tiny-aya-global"


def load_split(split: str, n: int | None = None) -> list[dict]:
    rows = [json.loads(x) for x in (ROOT / "data" / "tasks" / f"{split}.jsonl").open()]
    return rows[:n] if n else rows


def precision_kwargs() -> dict:
    """bf16 where supported (L40S); fp16 on T4."""
    import torch

    # TRL's configs default bf16=True, so always set both flags explicitly.
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return {"bf16": True, "fp16": False}
    if torch.cuda.is_available():
        return {"bf16": False, "fp16": True}
    return {"bf16": False, "fp16": False, "use_cpu": True}


def peak_mem() -> str:
    import torch

    if not torch.cuda.is_available():
        return "peak GPU mem n/a"
    return f"peak GPU mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
