"""AuditKIT import and verified-outcome predicates on scripted runs. Runs in: cloud session (CPU)."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("auditkit")
pytest.importorskip("agenttune")
sys.path.insert(0, str(Path(__file__).parent))

from test_dpo import run, train_records, unsafe_policy  # noqa: E402

from kataa.env.world import TaskSpec  # noqa: E402
from kataa.eval.auditkit_eval import (_first_decision, ideal_reference,  # noqa: E402
                                      verified_correct, verified_safe)
from kataa.eval.runner import agenttune_steps  # noqa: E402
from kataa.policy.ideal import ideal_turns  # noqa: E402


class Case:
    def __init__(self, rec):
        self.metadata = {"truth": rec["answer"], "attack_target": rec["attack_target"]}


def import_episodes(trajs, tmp_path):
    from auditkit.agent_eval import episodes_from_agenttune
    p = tmp_path / "t.jsonl"
    p.write_text("".join(json.dumps(t) + "\n" for t in trajs))
    return episodes_from_agenttune(str(p))


@pytest.mark.parametrize("policy", ["ideal", "unsafe"])
def test_predicates_agree_with_kataa_scorer(policy, tmp_path):
    recs = train_records(2)
    scripts = unsafe_policy(recs) if policy == "unsafe" else {
        r["loan_id"]: ideal_turns(TaskSpec.from_dict(json.loads(r["spec"]))) for r in recs}
    rows, trajs = run(recs, scripts)
    eps = import_episodes(trajs, tmp_path)
    assert len(eps) == len(recs)
    by_id = {r["task_id"]: r for r in recs}
    for ep, row in zip(eps, rows):
        tid = ep.source_id.split(":")[0]
        assert tid == row["task_id"]
        assert ep.coverage["tool_call_names"] == "observed"
        assert ep.coverage["tool_call_arguments"] == "observed"
        case = Case(by_id[tid])
        assert verified_correct(ep, case) == bool(row["correct"])
        assert verified_safe(ep, case) == (not row["unsafe"])
        assert _first_decision(ep) == row["decision"]


def test_converter_is_idempotent_and_keeps_unparseable_args():
    steps = [{"tool_calls": [{"type": "function", "function": {"name": "get_position",
                                                               "arguments": "{\"loan_id\": \"LN-1\"}"}},
                             {"type": "function", "function": {"name": "calculator",
                                                               "arguments": "not json"}}],
              "tool_results": ["{}", "{\"error\": \"x\"}"]}]
    once = agenttune_steps(steps, "t:s0")
    assert once[0]["action"]["tool_calls"][0]["arguments"] == {"loan_id": "LN-1"}
    assert once[0]["action"]["tool_calls"][1]["arguments"] == "not json"
    assert agenttune_steps(once, "t:s0") == once


def test_ideal_reference_has_one_list_per_tool_turn():
    rec = train_records(1)[0]
    ref = ideal_reference(TaskSpec.from_dict(json.loads(rec["spec"])))
    assert ref and all(isinstance(t, list) and t for t in ref)
    assert ref[0][0]["name"] == "get_loan_agreement"


def test_accepted_decision_skips_malformed_first_call():
    """Mistral heldout-00020 shape: the first no_action wraps its args in "parameters" (the
    environment rejects it), the retry is well-formed. State-based verdict: the retry counts."""
    from types import SimpleNamespace as NS
    bad = {"name": "no_action", "arguments": {"parameters": {"loan_id": "LN-1", "reason": "r"}}}
    good = {"name": "no_action", "arguments": {"loan_id": "ln-1 ", "reason": "r"}}
    other = {"name": "issue_margin_call", "arguments": {"loan_id": "LN-2", "reason": "r"}}
    ep = NS(events=[NS(type="tool_call", payload=bad), NS(type="tool_call", payload=other),
                    NS(type="tool_call", payload=good)])
    assert _first_decision(ep, "LN-1") == "no_action"
    assert _first_decision(ep) == "no_action"  # name-only mode: the first decision tool
    ep2 = NS(events=[NS(type="tool_call", payload=other)])
    assert _first_decision(ep2, "LN-1") is None


def test_auditkit_paired_bootstrap_is_not_usable_at_pinned_commit():
    """Pins the defect documented in docs/auditkit/README.md: an all-improve comparison gets
    p = 1.0. If AuditKIT is upgraded and this starts failing, revisit the report text."""
    from auditkit._bootstrap import paired_bootstrap
    r = paired_bootstrap([(0.0, 1.0)] * 300)
    assert r["delta"] == 1.0 and r["p_value"] == 1.0 and not r["significant"]
