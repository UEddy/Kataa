"""Ways of writing percentages in contract text, so limits must be read rather than parsed."""

from __future__ import annotations

import random

_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
          "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
          "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def int_words(n: int) -> str:
    if n < 20:
        return _UNITS[n]
    if n < 100:
        t, u = divmod(n, 10)
        return _TENS[t] + ("" if u == 0 else "-" + _UNITS[u])
    if n == 100:
        return "one hundred"
    raise ValueError(n)


def pct_words(p: float) -> str:
    """72.5 -> 'seventy-two and one-half'."""
    whole = int(p)
    frac = round(p - whole, 2)
    if frac == 0:
        return int_words(whole)
    if frac == 0.5:
        return f"{int_words(whole)} and one-half"
    if frac == 0.25:
        return f"{int_words(whole)} and one-quarter"
    if frac == 0.75:
        return f"{int_words(whole)} and three-quarters"
    raise ValueError(p)


def _num(p: float) -> str:
    return f"{p:g}"


def fmt_pct(p: float, rng: random.Random) -> str:
    """Render a percentage (e.g. 72.5) in one of several contract styles."""
    styles = [
        lambda: f"{_num(p)}%",
        lambda: f"{_num(p)} per cent.",
        lambda: f"{pct_words(p)} per cent. ({_num(p)}%)",
        lambda: f"{pct_words(p)} percent ({_num(p)}%)",
        lambda: f"{_num(p)} percent",
        lambda: f"a ratio of {p / 100:.4g}",
    ]
    return rng.choice(styles)()


def fmt_hours(h: int, rng: random.Random) -> str:
    styles = [lambda: f"{h} hours", lambda: f"{int_words(h)} ({h}) hours"]
    if h % 24 == 0:
        d = h // 24
        styles.append(lambda: f"{int_words(d)} ({d}) day{'s' if d > 1 else ''}"
                      if d > 1 else "one (1) day")
    return rng.choice(styles)()
