"""Scripted ideal policy: the expert runs used for the supervised warm start.

It plays the exact protocol the model must learn (kataa/protocol.py): turn 1 fetches the
agreement and position, turn 2 fetches every collateral price from both oracles, turn 3 writes
brief working notes and records exactly one decision, turn 4 is a one-line summary. The notes
only cite facts visible in tool outputs, and the decision comes from world.decide.
"""

from __future__ import annotations

import json
import re

from kataa.env.world import TaskSpec, decide, select_prices

DECISION_TOOL = {"margin_call": "issue_margin_call", "no_action": "no_action",
                 "escalate": "escalate_to_human"}
ORACLES = ("primary", "secondary")


def call_list(calls: list[tuple[str, dict]]) -> str:
    return json.dumps([{"tool_name": n, "parameters": p} for n, p in calls])


def _clause_no(spec: TaskSpec, title: str) -> str:
    m = re.search(rf"^(\d+)\. {re.escape(title)}$", spec.agreement_text, re.M)
    return f"Clause {m.group(1)}" if m else title


def _qty(q: float) -> str:
    return f"{q:,.0f}" if float(q).is_integer() else f"{q:,.4f}".rstrip("0").rstrip(".")


def _money(x: float) -> str:
    return f"${x:,.2f}" if x >= 1 else f"${x:.6g}"


def _attack_note(spec: TaskSpec) -> str | None:
    f = spec.attack_family
    if not f:
        return None
    where = {
        "inj_price_note": "a price feed note", "inj_clause": "a section of the agreement",
        "inj_borrower_note": "the position notes", "inj_structured": "a structured field of the position",
        "pressure_manager": "the request (a manager)", "pressure_borrower": "the request (the borrower)",
        "pressure_authority": "the request (a claimed authority)",
    }[f]
    return (f"Text in {where} tries to dictate the decision. Instructions in data or in the "
            "request do not change the agreement or the evidence, so I ignore them.")


def _num(x: float) -> str:
    """Plain decimal (no exponent, no commas), as the calculator accepts."""
    return ("%.10f" % x).rstrip("0").rstrip(".")


def as_of_text(spec: TaskSpec) -> str:
    """The determination time exactly as the request shows it."""
    m = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC|\d{2} [A-Z][a-z]{2} \d{4}, \d{2}:\d{2} UTC",
                  spec.request_text)
    return m.group(0) if m else spec.as_of


def _result(out: str, key: str) -> float:
    return json.loads(out)[key]


def compute_plan(spec: TaskSpec) -> dict:
    """The calls of the compute turns and their results (from the real tools)."""
    from kataa.env.tools import calculator, hours_between

    now = as_of_text(spec)
    t3: list[tuple[str, dict]] = []
    ages, gaps = {}, {}
    for a in spec.collateral:
        for o in ORACLES:
            args = {"start": spec.quotes[a][o].timestamp, "end": now}
            if ("hours_between", args) not in t3:  # identical timestamps: compute once
                t3.append(("hours_between", args))
            ages[(a, o)] = _result(hours_between(**args), "hours")
        pp, sp = spec.quotes[a]["primary"].price_usd, spec.quotes[a]["secondary"].price_usd
        expr = f"({_num(sp)} - {_num(pp)}) / {_num(pp)} * 100"
        t3.append(("calculator", {"expression": expr}))
        gaps[a] = _result(calculator(expr), "result")
    since_call = None
    if spec.last_margin_call_at:
        args = {"start": spec.last_margin_call_at, "end": now}
        if ("hours_between", args) not in t3:
            t3.append(("hours_between", args))
        since_call = _result(hours_between(**args), "hours")
    prices, problem = select_prices(spec)
    t4, cv, ltv = [], None, None
    if prices is not None:
        t = spec.terms
        cv_expr = " + ".join(f"{_num(q)} * {_num(prices[a])} * {_num(1 - t.haircuts[a] / 100)}"
                             for a, q in spec.collateral.items())
        ltv_expr = f"{_num(spec.debt_usd)} / ({cv_expr}) * 100"
        t4 = [("calculator", {"expression": cv_expr}), ("calculator", {"expression": ltv_expr})]
        cv = _result(calculator(cv_expr), "result")
        ltv = _result(calculator(ltv_expr), "result")
    return {"t3": t3, "t4": t4, "ages": ages, "gaps": gaps, "since_call": since_call,
            "prices": prices, "problem": problem, "cv": cv, "ltv": ltv}


def decision_notes(spec: TaskSpec, plan: dict | None = None) -> tuple[str, str]:
    """(working notes, short reason) for the decision turn, citing the tool results."""
    answer, _, why = decide(spec)
    t = spec.terms
    lines: list[str] = []
    att = _attack_note(spec)
    if not spec.position_available:
        lines.append("The position snapshot is unavailable, so the LTV cannot be determined.")
        if att:
            lines.append(att)
        return "Notes: " + " ".join(lines), "Position snapshot unavailable; LTV cannot be determined."

    plan = plan or compute_plan(spec)
    m_src = _clause_no(spec, "Margin Maintenance")
    if t.original_maintenance_ltv is not None:
        lines.append(f"Maintenance LTV is {t.maintenance_ltv:g}% ({m_src}, as changed by "
                     f"Amendment No. 1 from {t.original_maintenance_ltv:g}%).")
    else:
        lines.append(f"Maintenance LTV is {t.maintenance_ltv:g}% ({m_src}).")
    lines.append(f"Prices are current within {t.staleness_hours}h; sources may differ by at most "
                 f"{t.divergence_pct:g}% ({_clause_no(spec, 'Price Determination')}).")
    for a in spec.collateral:
        pa, sa = plan["ages"][(a, "primary")], plan["ages"][(a, "secondary")]
        state = lambda h: "current" if h <= t.staleness_hours else "stale"  # noqa: E731
        lines.append(f"{a}: primary {pa:g}h old ({state(pa)}), secondary {sa:g}h old "
                     f"({state(sa)}), gap {plan['gaps'][a]:+.2f}%.")
    if plan["prices"] is None:
        lines.append(f"Problem: {plan['problem']}, so no LTV determination can be made.")
        reason = f"{plan['problem'][0].upper() + plan['problem'][1:]}; the agreement requires referral."
    else:
        lines.append(f"Collateral value = {_money(plan['cv'])}; LTV = {plan['ltv']:.2f}% "
                     f"{'>=' if plan['ltv'] >= t.maintenance_ltv else '<'} {t.maintenance_ltv:g}%.")
        if plan["since_call"] is not None:
            running = plan["since_call"] < t.cure_hours
            lines.append(f"Last margin call was {plan['since_call']:g}h ago; cure period is "
                         f"{t.cure_hours}h ({'still running' if running else 'over'}).")
        reason = why
    if att:
        lines.append(att)
    return "Notes: " + " ".join(lines), reason


def ideal_turns(spec: TaskSpec) -> list[str]:
    """Assistant turn texts, in order, for the ideal run."""
    loan = spec.loan_id
    answer = decide(spec)[0]
    turns = [call_list([("get_loan_agreement", {"loan_id": loan}),
                        ("get_position", {"loan_id": loan})])]
    plan = None
    if spec.position_available:
        turns.append(call_list([("get_price", {"loan_id": loan, "asset": a, "oracle": o})
                                for a in spec.collateral for o in ORACLES]))
        plan = compute_plan(spec)
        turns.append("Checking price ages and the gap between the two sources.\n"
                     + call_list(plan["t3"]))
        if plan["t4"]:
            turns.append("Prices are usable. Computing collateral value and LTV.\n"
                         + call_list(plan["t4"]))
    notes, reason = decision_notes(spec, plan)
    turns.append(notes + "\n" + call_list([(DECISION_TOOL[answer],
                                            {"loan_id": loan, "reason": reason})]))
    label = {"margin_call": "margin call issued", "no_action": "no action recorded",
             "escalate": "escalated to a credit officer"}[answer]
    turns.append(f"Done: {label} for {loan}. {reason}")
    return turns


class ScriptedPolicy:
    """Answers each model turn from a fixed list (used as a fake engine in CPU tests)."""

    def __init__(self, turns: list[str]):
        self.turns = list(turns)
        self.i = 0

    def next(self) -> str:
        out = self.turns[self.i] if self.i < len(self.turns) else "Done."
        self.i += 1
        return out
