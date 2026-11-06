"""The batched eval loop, driven by a fake LLM. Runs in: cloud session (CPU)."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("agenttune")
sys.path.insert(0, str(Path(__file__).parent))

from kataa.env.world import TaskSpec  # noqa: E402
from kataa.eval.runner import run_episodes, score, summarize, to_agenttune_jsonl  # noqa: E402
from kataa.policy.ideal import call_list, ideal_turns  # noqa: E402
from kataa.rewards.rewards import R_CORRECT, R_FORMAT  # noqa: E402
from kataa.train.sft import ideal_conversation  # noqa: E402
from test_e2e_agenttune import sample_records  # noqa: E402


def scripted(scripts):
    def generate(convs):
        out = []
        for c in convs:
            user = next(m["content"] for m in c if m["role"] == "user")
            loan = next(k for k in scripts if k in user)
            k = sum(m["role"] == "assistant" for m in c)
            out.append(scripts[loan][k] if k < len(scripts[loan]) else "Done.")
        return out
    return generate


def test_ideal_policy_matches_training_protocol():
    recs = sample_records()
    eps = run_episodes(scripted({r["loan_id"]: ideal_turns(TaskSpec.from_dict(json.loads(r["spec"])))
                                 for r in recs}), recs, n_samples=2)
    rows = score(eps)
    assert all(r["hardened"] == pytest.approx(R_CORRECT + R_FORMAT) for r in rows)
    s = summarize(rows)
    assert s["accuracy"] == 1 and s["unsafe"] == 0 and s["naive_spread_groups"] == 0
    for e in eps:  # same conversation as the warm-start data and the GRPO rollout
        ours = ideal_conversation(e.record)
        assert [(m["role"], m["content"]) for m in e.conv] == \
               [(m["role"], m["content"]) for m in ours]
    row = to_agenttune_jsonl(eps[0])
    assert {"task", "steps", "final_response", "metadata"} <= set(row)


def test_gaming_and_turn_limit():
    recs = sample_records()[:5]
    esc = {r["loan_id"]: [call_list([("escalate_to_human",
                                      {"loan_id": r["loan_id"], "reason": "x"})])] for r in recs}
    s = summarize(score(run_episodes(scripted(esc), recs)))
    assert s["evidence_ok"] == 0 and s["decided_escalate"] == 1
    loop = {r["loan_id"]: [call_list([("get_position", {"loan_id": r["loan_id"]})])] * 50
            for r in recs}
    eps = run_episodes(scripted(loop), recs, max_tool_turns=6)
    assert all(e.tool_iters == 6 for e in eps)
