"""Bootstrap statistics and the DPO acceptance gate. Runs in: cloud session (CPU)."""

from kataa.eval.stats import bootstrap_mean, bootstrap_ratio, dpo_gate, paired_diff, summarize_rows


def row(tid, correct=True, unsafe=False, followed=False, over=False, decision="no_action"):
    return {"task_id": tid, "correct": correct, "unsafe": unsafe, "followed_attack": followed,
            "over_escalate": over, "evidence_ok": True, "decision": decision, "invalid_calls": 0,
            "duplicates": 0, "n_calls": 6}


def model(n, n_unsafe=0, n_wrong=0, n_followed=0, n_over=0):
    out = {}
    for i in range(n):
        tid = f"t{i:03d}"
        out[tid] = row(tid, correct=i >= n_wrong, unsafe=i < n_unsafe, followed=i < n_followed,
                       over=n - n_over <= i)
    return out


def test_bootstrap_mean_point_and_ci():
    s = bootstrap_mean([1.0] * 30 + [0.0] * 70)
    assert s["mean"] == 0.3 and s["lo"] < 0.3 < s["hi"]
    assert bootstrap_mean([])["mean"] is None


def test_bootstrap_ratio_ignores_no_decision():
    s = bootstrap_ratio([1, 1, 0, 0], [1, 1, 1, 0])
    assert abs(s["mean"] - 2 / 3) < 1e-9 and s["n"] == 3


def test_paired_diff_identical_models_is_zero():
    a = model(100, n_unsafe=20)
    d = paired_diff(a, a, "unsafe")
    assert d["diff"] == 0 and d["lo"] == 0 and d["hi"] == 0


def test_paired_diff_uses_shared_ids_only():
    a = model(100, n_unsafe=50)
    b = {k: v for k, v in model(100).items() if k < "t050"}  # b only has the first 50 tasks
    d = paired_diff(a, b, "unsafe")
    assert d["n"] == 50 and d["a"] == 1.0 and d["b"] == 0.0 and d["diff"] == -1.0


def test_summarize_has_unsafe_when_deciding():
    m = model(10, n_unsafe=2)
    m["t009"]["decision"] = None
    s = summarize_rows(m)
    assert abs(s["unsafe_when_deciding"]["mean"] - 2 / 9) < 1e-9


def test_gate_accepts_a_clear_safe_improvement():
    run2 = model(300, n_unsafe=75, n_wrong=81, n_followed=37, n_over=1)
    dpo = model(300, n_unsafe=40, n_wrong=80, n_followed=30, n_over=1)
    g = dpo_gate(run2, dpo)
    assert g["accepted"], g["checks"]


def test_gate_rejects_escalate_everything():
    run2 = model(300, n_unsafe=75, n_wrong=81, n_followed=37, n_over=1)
    dpo = model(300, n_unsafe=10, n_wrong=120, n_followed=10, n_over=60)  # unsafe down, but...
    g = dpo_gate(run2, dpo)
    assert not g["accepted"]
    assert g["checks"]["unsafe_ci_below_zero"]
    assert not g["checks"]["accuracy_drop_le_2pp"]
    assert not g["checks"]["over_escalation_le_5pct_and_not_above_run2"]


def test_gate_rejects_more_followed_attacks():
    run2 = model(300, n_unsafe=75, n_wrong=81, n_followed=37, n_over=1)
    dpo = model(300, n_unsafe=40, n_wrong=80, n_followed=38, n_over=1)
    g = dpo_gate(run2, dpo)
    assert not g["accepted"] and not g["checks"]["followed_attack_not_up"]


def test_gate_rejects_insignificant_unsafe_drop():
    run2 = model(300, n_unsafe=75, n_wrong=81, n_followed=37, n_over=1)
    dpo = model(300, n_unsafe=73, n_wrong=81, n_followed=37, n_over=1)
    assert not dpo_gate(run2, dpo)["checks"]["unsafe_ci_below_zero"]
