"""Replay of real daily prices (Coin Metrics community data, CC BY-NC 4.0; see DATA.md)."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path

PRICES_CSV = Path(__file__).resolve().parents[2] / "data" / "prices" / "coinmetrics_daily.csv"

# Coin Metrics' daily PriceUSD is the close of the UTC day; we stamp it at 23:59 UTC that day.
CLOSE_TIME = time(23, 59, tzinfo=timezone.utc)

# Real stress days in the snapshot (largest one-day BTC/ETH moves), oversampled by the generator.
STRESS_DAYS = [
    "2020-03-12", "2020-03-13", "2021-05-19", "2021-01-21",
    "2022-06-13", "2022-11-09", "2019-06-27", "2021-05-24",
]


class PriceBook:
    """Daily closes per asset, keyed by ISO date."""

    def __init__(self, path: Path = PRICES_CSV):
        self._px: dict[str, dict[str, float]] = defaultdict(dict)
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                self._px[row["asset"]][row["date"]] = float(row["price_usd"])
        self.assets = sorted(self._px)
        self.days = sorted(set().union(*[set(v) for v in self._px.values()]))

    def close(self, asset: str, day: str) -> float:
        return self._px[asset][day]

    def has(self, asset: str, day: str) -> bool:
        return day in self._px.get(asset, {})


def close_timestamp(day: str) -> datetime:
    return datetime.combine(date.fromisoformat(day), CLOSE_TIME)


def shift_day(day: str, days: int) -> str:
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


def iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@lru_cache(maxsize=1)
def default_book() -> PriceBook:
    return PriceBook()
