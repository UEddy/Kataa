"""Global loan registry the tools read from.

AgentTune calls tools as plain functions with only the model's arguments, so there is no
per-rollout state: every task has a globally unique loan_id and the tools look it up here.
"""

from __future__ import annotations

import json
from pathlib import Path

from .world import TaskSpec

TASK_DIR = Path(__file__).resolve().parents[2] / "data" / "tasks"
_REGISTRY: dict[str, TaskSpec] = {}


def register(spec: TaskSpec) -> None:
    if spec.loan_id in _REGISTRY and _REGISTRY[spec.loan_id].task_id != spec.task_id:
        raise ValueError(f"duplicate loan_id {spec.loan_id}")
    _REGISTRY[spec.loan_id] = spec


def load_records(records) -> int:
    """Register specs from dataset rows (dicts with a JSON `spec` column)."""
    n = 0
    for r in records:
        register(TaskSpec.from_dict(json.loads(r["spec"])))
        n += 1
    return n


def load_dir(path: Path = TASK_DIR) -> int:
    n = 0
    for f in sorted(Path(path).glob("*.jsonl")):
        with open(f) as fh:
            n += load_records(json.loads(line) for line in fh if line.strip())
    return n


def get(loan_id: str) -> TaskSpec | None:
    if not _REGISTRY:
        load_dir()
    return _REGISTRY.get(str(loan_id).strip())


def clear() -> None:
    _REGISTRY.clear()
