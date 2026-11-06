"""Write the scripted ideal runs (warm-start data). Runs in: cloud session.

    python scripts/build_sft.py            # data/sft/{train,val}.jsonl

Each line: task_id, answer, condition, attack_family, and `messages`, the full conversation in
the exact format the GRPO rollout produces (see tests/test_e2e_agenttune.py).
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kataa.rewards import hardened_reward  # noqa: E402
from kataa.rewards.rewards import R_CORRECT, R_FORMAT  # noqa: E402
from kataa.train.sft import ideal_conversation  # noqa: E402


def main() -> int:
    out_dir = ROOT / "data" / "sft"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        rows = [json.loads(x) for x in (ROOT / "data" / "tasks" / f"{split}.jsonl").open()]
        convs = [ideal_conversation(r) for r in rows]
        scores = hardened_reward(completions=None, conversations=convs,
                                 answer=[r["answer"] for r in rows],
                                 spec=[r["spec"] for r in rows],
                                 attack_target=[r["attack_target"] for r in rows])
        assert all(abs(s - (R_CORRECT + R_FORMAT)) < 1e-9 for s in scores), split
        with open(out_dir / f"{split}.jsonl", "w") as f:
            for r, c in zip(rows, convs):
                f.write(json.dumps({"task_id": r["task_id"], "answer": r["answer"],
                                    "condition": r["condition"],
                                    "attack_family": r["attack_family"], "messages": c}) + "\n")
        print(split, len(rows), dict(Counter(r["answer"] for r in rows)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
