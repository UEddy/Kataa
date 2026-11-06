"""The agent's tools: plain functions with Google-style docstrings (AgentTune builds the schemas
with transformers.utils.get_json_schema). Each returns a JSON string. Unknown IDs return an error
object rather than raising, so the model sees a clean message.

get_price takes loan_id as well as asset and oracle: tools are stateless, and the loan fixes which
determination time (and therefore which price snapshot) applies.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
from datetime import datetime, timezone

from . import registry

ORACLE_NAMES = ("primary", "secondary")


def _err(msg: str) -> str:
    return json.dumps({"error": msg})


def _unknown(loan_id) -> str:
    return _err(f"unknown loan_id {loan_id!r}")


def get_loan_agreement(loan_id: str) -> str:
    """Return the full text of the loan agreement, including any amendments.

    Args:
        loan_id: The loan identifier, for example "LN-1A2B3C".
    """
    spec = registry.get(loan_id)
    if spec is None:
        return _unknown(loan_id)
    return json.dumps({"loan_id": spec.loan_id, "agreement": spec.agreement_text})


def get_position(loan_id: str) -> str:
    """Return the loan's current position: outstanding debt, collateral holdings, the time of the
    most recent margin call, and any notes on file.

    Args:
        loan_id: The loan identifier, for example "LN-1A2B3C".
    """
    spec = registry.get(loan_id)
    if spec is None:
        return _unknown(loan_id)
    if not spec.position_available:
        return _err(f"position snapshot unavailable for {spec.loan_id}: the custody feed "
                    "returned no record")
    out = {
        "loan_id": spec.loan_id,
        "borrower": spec.borrower,
        "outstanding_debt_usd": spec.debt_usd,
        "collateral": [{"asset": a, "quantity": q} for a, q in spec.collateral.items()],
        "last_margin_call_at": spec.last_margin_call_at,
    }
    if spec.borrower_notes:
        out["notes"] = spec.borrower_notes
    if spec.position_extra:
        out.update(spec.position_extra)
    return json.dumps(out)


def get_price(loan_id: str, asset: str, oracle: str) -> str:
    """Return the latest price of a collateral asset from one price oracle, with its timestamp.

    Args:
        loan_id: The loan identifier, for example "LN-1A2B3C".
        asset: The asset ticker, for example "BTC".
        oracle: Which price source to query: "primary" or "secondary".
    """
    spec = registry.get(loan_id)
    if spec is None:
        return _unknown(loan_id)
    a = str(asset).strip().upper()
    o = str(oracle).strip().lower()
    if o not in ORACLE_NAMES:
        return _err(f"unknown oracle {oracle!r}; use 'primary' or 'secondary'")
    if a not in spec.quotes:
        return _err(f"{asset!r} is not collateral for {spec.loan_id}")
    q = spec.quotes[a][o]
    out = {"asset": a, "oracle": o, "price_usd": q.price_usd, "timestamp": q.timestamp}
    if q.note:
        out["note"] = q.note
    return json.dumps(out)


def _decision(loan_id: str, decision: str, reason: str) -> str:
    spec = registry.get(loan_id)
    if spec is None:
        return _unknown(loan_id)
    return json.dumps({"status": "recorded", "loan_id": spec.loan_id, "decision": decision,
                       "reason": str(reason)})


def issue_margin_call(loan_id: str, reason: str) -> str:
    """Issue a margin call to the borrower. This is a final decision for the loan.

    Args:
        loan_id: The loan identifier, for example "LN-1A2B3C".
        reason: A short justification citing the evidence.
    """
    return _decision(loan_id, "margin_call", reason)


def escalate_to_human(loan_id: str, reason: str) -> str:
    """Refer the loan to a human credit officer without deciding. This is a final decision.

    Args:
        loan_id: The loan identifier, for example "LN-1A2B3C".
        reason: A short justification citing the evidence.
    """
    return _decision(loan_id, "escalate", reason)


def no_action(loan_id: str, reason: str) -> str:
    """Record that no margin action is required for the loan. This is a final decision.

    Args:
        loan_id: The loan identifier, for example "LN-1A2B3C".
        reason: A short justification citing the evidence.
    """
    return _decision(loan_id, "no_action", reason)


# ── Utility tools (no loan state) ────────────────────────────────────────────

CALC_MAX_LEN = 200
CALC_TIMEOUT_S = 2.0
_CALC_OK = re.compile(r"^[0-9+\-*/().%\s]+$")
_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def calculator(expression: str) -> str:
    """Evaluate a purely numeric arithmetic expression and return the result.

    Args:
        expression: Digits, spaces and + - * / ( ) . % only, at most 200 characters,
            for example "193700 / (15.0773 * 9335.18 * 0.9) * 100".
    """
    e = str(expression)
    if len(e) > CALC_MAX_LEN:
        return _err(f"expression longer than {CALC_MAX_LEN} characters")
    if not _CALC_OK.fullmatch(e) or "**" in e or not re.search(r"\d", e):
        return _err("only digits, spaces and + - * / ( ) . % are allowed")
    # AgentTune's built-in calculator (asteval) evaluates the already-validated expression.
    from agenttune.agentic.tools.builtin.finqa_tool import calculator as _agenttune_calc

    fut = _POOL.submit(_agenttune_calc, e)
    try:
        out = fut.result(timeout=CALC_TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        return _err("calculation timed out")
    if isinstance(out, str) or out is None:
        return _err(f"could not evaluate {e!r}")
    return json.dumps({"expression": e, "result": round(float(out), 6)})


_TS_FORMATS = ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%MZ", "%Y-%m-%d %H:%M UTC",
               "%Y-%m-%d %H:%M:%S UTC", "%Y-%m-%d %H:%M", "%d %b %Y, %H:%M UTC", "%d %b %Y %H:%M UTC")


def _parse_ts(s: str) -> datetime | None:
    s = str(s).strip()
    for fmt in _TS_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
        except ValueError:
            continue
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def hours_between(start: str, end: str) -> str:
    """Return the number of hours from `start` to `end` (negative if `end` is earlier).

    Args:
        start: A UTC time, for example "2022-11-09T23:59:00Z" or "2022-11-10 06:00 UTC".
        end: A UTC time in the same formats, for example the determination time.
    """
    a, b = _parse_ts(start), _parse_ts(end)
    if a is None or b is None:
        return _err("could not parse a time; use e.g. 2022-11-09T23:59:00Z or 2022-11-10 06:00 UTC")
    return json.dumps({"start": str(start), "end": str(end),
                       "hours": round((b - a).total_seconds() / 3600.0, 2)})


EVIDENCE = [get_loan_agreement, get_position, get_price]
UTILITY = [calculator, hours_between]
DECISIONS = [issue_margin_call, no_action, escalate_to_human]
TOOLS = EVIDENCE + UTILITY + DECISIONS
