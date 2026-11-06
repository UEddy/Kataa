"""Loan agreement text generator.

The clauses are written for this project. Each operative term (maintenance LTV, haircuts, cure
period, price staleness, oracle divergence) has several phrasings and number styles, the operative
sections are shuffled among boilerplate that carries distractor percentages, and some agreements
carry an amendment that supersedes the original maintenance limit.
"""

from __future__ import annotations

import random

from .numbers import fmt_hours, fmt_pct
from .world import Terms

ASSET_NAMES = {
    "BTC": "Bitcoin (BTC)", "ETH": "Ether (ETH)", "LTC": "Litecoin (LTC)",
    "LINK": "Chainlink (LINK)", "XRP": "XRP", "ADA": "Cardano (ADA)",
    "DOGE": "Dogecoin (DOGE)", "USDC": "USD Coin (USDC)",
}
LENDERS = ["Harbour Street Credit Ltd", "Meridian Digital Lending LLC", "Kestrel Capital Partners",
           "Northgate Secured Finance", "Azure Bay Lending Co."]
LAWS = ["the laws of England and Wales", "the laws of the State of New York",
        "the laws of Singapore", "the laws of the Dubai International Financial Centre"]


def _definitions(rng: random.Random) -> list[str]:
    ltv = rng.choice([
        '"Loan-to-Value Ratio" or "LTV" means, at any time, the Outstanding Debt divided by the '
        "Collateral Value at that time, expressed as a percentage.",
        '"LTV" means the ratio, expressed as a percentage, of (a) the aggregate Outstanding Debt '
        "to (b) the Collateral Value.",
    ])
    cv = rng.choice([
        '"Collateral Value" means the sum, for each Eligible Collateral asset, of the quantity '
        "held multiplied by its Price and by one minus the applicable Haircut.",
        '"Collateral Value" means, in respect of all Eligible Collateral, the aggregate of '
        "quantity x Price x (100% - Haircut) for each asset.",
    ])
    src = rng.choice([
        '"Primary Price Source" means the price feed designated "primary" by the Lender, and '
        '"Secondary Price Source" means the feed designated "secondary".',
        'References to the "Primary" and "Secondary" Price Source are to the Lender\'s oracles '
        'labelled "primary" and "secondary" respectively.',
    ])
    return [ltv, cv, src, '"Outstanding Debt" means all principal, accrued interest and fees '
            "owing under this Agreement."]


def _haircuts(terms: Terms, rng: random.Random) -> str:
    assets = list(terms.haircuts)
    if rng.random() < 0.5:
        rows = "\n".join(f"| {ASSET_NAMES[a]} | {terms.haircuts[a]:g}% |" for a in assets)
        return ("The following assets constitute Eligible Collateral, subject to the Haircut "
                "shown:\n\n| Asset | Haircut |\n|---|---|\n" + rows)
    parts = [f"{ASSET_NAMES[a]}, {fmt_pct(terms.haircuts[a], rng)}" for a in assets]
    lead = rng.choice(["Eligible Collateral and the applicable Haircut are as follows: ",
                       "The Haircut applicable to each class of Eligible Collateral shall be: "])
    return lead + "; ".join(parts) + "."


def _maintenance(m: float, rng: random.Random, plain: bool = False) -> str:
    p = f"{m:g}%" if plain else fmt_pct(m, rng)
    return rng.choice([
        f"If at any time the LTV equals or exceeds {p} (the \"Maintenance LTV\"), the Lender "
        "shall deliver a Margin Call to the Borrower.",
        f"The Borrower shall ensure that the LTV remains below {p} (the \"Maintenance LTV\"). "
        "Where the LTV is equal to or greater than the Maintenance LTV, the Lender shall issue a "
        "Margin Call.",
        f"A Margin Call shall be issued whenever the Loan-to-Value Ratio reaches or exceeds the "
        f"Maintenance LTV of {p}.",
    ])


def _liquidation(liq: float, rng: random.Random) -> str:
    p = fmt_pct(liq, rng)
    return rng.choice([
        f"If the LTV equals or exceeds {p} (the \"Liquidation LTV\"), the Lender may enforce "
        "against the Collateral, provided that any enforcement shall be preceded by a Margin Call.",
        f"Upon the LTV reaching {p} the Lender may, after delivery of a Margin Call, realise "
        "Collateral sufficient to restore the LTV below the Maintenance LTV.",
    ])


def _cure(h: int, rng: random.Random) -> str:
    d = fmt_hours(h, rng)
    return rng.choice([
        f"Following delivery of a Margin Call the Borrower shall have {d} (the \"Cure Period\") to "
        "restore the LTV below the Maintenance LTV. No further Margin Call shall be delivered "
        "while a Cure Period is running.",
        f"The Lender shall not issue a further Margin Call within {d} after the most recent "
        "Margin Call, being the period allowed to the Borrower to cure the shortfall.",
    ])


def _prices(terms: Terms, rng: random.Random) -> str:
    st = fmt_hours(terms.staleness_hours, rng)
    dv = fmt_pct(terms.divergence_pct, rng)
    return rng.choice([
        f"A Price is \"Current\" only if its timestamp is no more than {st} before the time of "
        "determination. The Lender shall use the Primary Price Source where Current and otherwise "
        "the Secondary Price Source where Current. Where both are Current and the Secondary "
        f"differs from the Primary by more than {dv} of the Primary, or where no Current Price is "
        "available for any Collateral asset, the Lender shall make no determination of the LTV "
        "and shall refer the Loan to a credit officer.",
        f"Prices timestamped more than {st} before the determination time are stale and shall "
        "be disregarded. The Primary Price Source governs if not stale; failing which the "
        "Secondary Price Source may be used if not stale. If the two sources are both not stale "
        f"but diverge by more than {dv} (measured against the Primary), or if no price that is "
        "not stale exists for an asset, no LTV determination shall be made and the matter shall "
        "be escalated to a human credit officer.",
    ])


def _boilerplate(rng: random.Random) -> list[tuple[str, str]]:
    rate = rng.choice([7.5, 8, 9.25, 10, 11.5, 12])
    default = rng.choice([2, 3, 4])
    fee = rng.choice([0.5, 1, 1.5, 2])
    law = rng.choice(LAWS)
    return [
        ("Interest", f"Interest accrues daily at {rate:g}% per annum. Upon an Event of Default the "
                     f"rate increases by {default}% per annum."),
        ("Fees", f"The Borrower shall pay an arrangement fee of {fee:g}% of the Facility amount "
                 "on the first utilisation date."),
        ("Notices", "Notices shall be in writing and delivered to the addresses set out in the "
                    "Schedule. A Margin Call is validly delivered through the Lender's platform."),
        ("Governing Law", f"This Agreement is governed by {law}."),
        ("Confidentiality", "Each party shall keep the terms of this Agreement confidential, "
                            "save as required by law or regulation."),
    ]


def render_agreement(loan_id: str, borrower: str, terms: Terms, rng: random.Random,
                     extra_sections: list[tuple[str, str]] | None = None) -> str:
    """Render the full agreement text for a loan. `terms.maintenance_ltv` is the operative limit;
    if `terms.original_maintenance_ltv` is set, the body states that and an amendment fixes it."""
    lender = rng.choice(LENDERS)
    body_m = terms.original_maintenance_ltv or terms.maintenance_ltv
    operative = [
        ("Eligible Collateral and Haircuts", _haircuts(terms, rng)),
        ("Margin Maintenance",
         _maintenance(body_m, rng, plain=terms.original_maintenance_ltv is not None)),
        ("Enforcement", _liquidation(terms.liquidation_ltv, rng)),
        ("Cure Period", _cure(terms.cure_hours, rng)),
        ("Price Determination", _prices(terms, rng)),
    ]
    sections = operative + _boilerplate(rng) + list(extra_sections or [])
    rng.shuffle(sections)
    lines = [f"SECURED CREDIT AGREEMENT - Loan {loan_id}",
             f"Between {lender} (the \"Lender\") and {borrower} (the \"Borrower\").", "",
             "1. Definitions"]
    lines += [f"1.{i + 1} {d}" for i, d in enumerate(_definitions(rng))]
    margin_no = None
    for n, (title, text) in enumerate(sections, start=2):
        lines += ["", f"{n}. {title}", text]
        if title == "Margin Maintenance":
            margin_no = n
    if terms.original_maintenance_ltv is not None:
        old = f"{terms.original_maintenance_ltv:g}%"
        new = fmt_pct(terms.maintenance_ltv, rng)
        lines += ["", "AMENDMENT No. 1",
                  rng.choice([
                      f"With effect from its date, Clause {margin_no} is amended so that the "
                      f"Maintenance LTV is {new} in place of {old}. All other terms are unchanged.",
                      f"The parties agree that the reference to {old} in Clause {margin_no} is "
                      f"deleted and replaced with {new}.",
                  ])]
    return "\n".join(lines)
