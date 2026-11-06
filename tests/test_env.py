"""Environment tests. Runs in: cloud session (CPU, no model)."""

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from kataa.env import registry, tools
from kataa.env.attacks import HELDOUT_FAMILIES, TRAIN_FAMILIES
from kataa.env.generate import SPLITS, generate, make_task, to_record
from kataa.env.prices import default_book
from kataa.env.world import (DECISIONS, TaskSpec, decide, hours_between)

TASK_DIR = Path(__file__).resolve().parents[1] / "data" / "tasks"


@pytest.fixture(scope="module")
def specs():
    return {s: generate(s) for s in SPLITS}


def test_generation_is_deterministic(specs):
    again = generate("val")
    assert [to_record(s) for s in again] == [to_record(s) for s in specs["val"]]


def test_committed_tasks_match_generator(specs):
    for split, ss in specs.items():
        lines = (TASK_DIR / f"{split}.jsonl").read_text().splitlines()
        assert [json.loads(line) for line in lines] == \
               [json.loads(json.dumps(to_record(s), sort_keys=True)) for s in ss], split


def test_loan_ids_unique_across_splits(specs):
    ids = [s.loan_id for ss in specs.values() for s in ss]
    assert len(ids) == len(set(ids))


def test_heldout_families_never_in_training(specs):
    for split in ("train", "val", "test"):
        fams = {s.attack_family for s in specs[split] if s.attack_family}
        assert fams <= set(TRAIN_FAMILIES)
        assert not fams & set(HELDOUT_FAMILIES)
    assert {s.attack_family for s in specs["heldout"]} == set(HELDOUT_FAMILIES)
    assert all(s.heldout for s in specs["heldout"])


def test_answer_balance(specs):
    for split, ss in specs.items():
        c = Counter(s.answer for s in ss)
        assert set(c) == set(DECISIONS), split
        assert max(c.values()) / len(ss) < 0.5, (split, c)  # no constant answer wins


def test_roundtrip_and_ground_truth(specs):
    for s in specs["train"]:
        s2 = TaskSpec.from_dict(json.loads(json.dumps(s.to_dict())))
        assert decide(s2)[0] == s.answer


def test_condition_semantics(specs):
    for s in specs["train"]:
        t = s.terms
        if s.condition in ("stale_both", "oracle_conflict", "missing_position"):
            assert s.answer == "escalate"
        if s.condition == "cure_active":
            assert s.answer == "no_action" and s.ltv_pct >= t.maintenance_ltv
            assert hours_between(s.as_of, s.last_margin_call_at) < t.cure_hours
        if s.condition == "cure_expired":
            assert s.answer == "margin_call"
        if s.ltv_pct is not None:  # never on a knife edge
            assert abs(s.ltv_pct - t.maintenance_ltv) >= 0.2


def test_prices_are_real_replays(specs):
    book = default_book()
    for s in specs["train"][:300]:
        for a, qs in s.quotes.items():
            p = qs["primary"]
            assert p.price_usd == book.close(a, p.source_day)


def test_agreement_states_operative_limit(specs):
    for s in specs["train"]:
        m = s.terms.maintenance_ltv
        text = s.agreement_text
        assert f"{m:g}" in text or f"{m / 100:.4g}" in text
        if s.terms.original_maintenance_ltv is not None:
            assert "AMENDMENT No. 1" in text


def _surface(s):
    """Everything the model can see for a task: the request and every evidence tool output."""
    parts = [s.request_text, tools.get_loan_agreement(s.loan_id), tools.get_position(s.loan_id)]
    parts += [tools.get_price(s.loan_id, a, o) for a in s.collateral
              for o in ("primary", "secondary")]
    return "\n".join(parts)


def test_attack_text_reaches_the_model(specs):
    from kataa.env.attacks import ACTION, TOOL_NAME
    markers = [p for ps in ACTION.values() for p in ps] + list(TOOL_NAME.values()) + \
              ["don't margin call me", "real person look at my loan"]
    registry.clear()
    for ss in specs.values():
        for s in ss:
            registry.register(s)
    for s in specs["train"] + specs["heldout"]:
        seen = _surface(s).replace("\\u2019", "'")
        hit = any(m.lower() in seen.lower() for m in markers)
        assert hit == (s.attack_family is not None), (s.task_id, s.attack_family)
    registry.clear()


def test_tools(specs):
    registry.clear()
    s = specs["test"][0]
    registry.register(s)
    assert "error" in json.loads(tools.get_position("LN-000000"))
    assert "error" in json.loads(tools.issue_margin_call("nope", "x"))
    a = next(iter(s.collateral))
    q = json.loads(tools.get_price(s.loan_id, a.lower(), "Primary"))
    assert q["price_usd"] == s.quotes[a]["primary"].price_usd
    assert "error" in json.loads(tools.get_price(s.loan_id, a, "tertiary"))
    assert "error" in json.loads(tools.get_price(s.loan_id, "ZZZ", "primary"))
    assert json.loads(tools.no_action(s.loan_id, "fine"))["status"] == "recorded"
    registry.clear()


def test_missing_position_tool_error(specs):
    registry.clear()
    s = next(x for x in specs["train"] if x.condition == "missing_position")
    registry.register(s)
    assert "unavailable" in json.loads(tools.get_position(s.loan_id))["error"]
    registry.clear()


def test_attacks_do_not_change_answer():
    book = default_book()
    for i in range(60):
        clean = make_task("probe", i, 99, 0.0, TRAIN_FAMILIES, book)
        attacked = make_task("probe", i, 99, 1.0, TRAIN_FAMILIES, book)
        # The same seed builds the same world; only the attack overlay differs.
        assert clean.answer == attacked.answer
        assert clean.debt_usd == attacked.debt_usd and clean.as_of == attacked.as_of
