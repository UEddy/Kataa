"""Fetch real daily crypto prices from the Coin Metrics community archive.

Runs in: cloud session. Source: https://github.com/coinmetrics/data (CC BY-NC 4.0),
pinned to one commit so the snapshot is reproducible. Writes a compact CSV with two real
PriceUSD per asset per day. (ReferenceRateUSD is only populated for the last few days in the
archive, so it is not used.) The second oracle in the environment is derived from this series;
see DATA.md.
Raw downloads go to data/raw/ (gitignored) and are treated as data only, never executed.
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.request
from pathlib import Path

COMMIT = "f1a36afb962731c387bb03982758ab0103063da5"
BASE = f"https://raw.githubusercontent.com/coinmetrics/data/{COMMIT}/csv/{{asset}}.csv"
ASSETS = ["btc", "eth", "ltc", "link", "xrp", "ada", "doge", "usdc"]
START, END = "2019-01-01", "2025-12-31"

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "prices" / "coinmetrics_daily.csv"


def fetch(asset: str) -> str:
    RAW.mkdir(parents=True, exist_ok=True)
    cached = RAW / f"{asset}_{COMMIT[:8]}.csv"
    if not cached.exists():
        with urllib.request.urlopen(BASE.format(asset=asset), timeout=60) as r:
            cached.write_bytes(r.read())
    return cached.read_text()


def main() -> int:
    rows = []
    for asset in ASSETS:
        reader = csv.DictReader(io.StringIO(fetch(asset)))
        n = 0
        for rec in reader:
            day = rec["time"]
            if not (START <= day <= END):
                continue
            px = rec.get("PriceUSD", "")
            if not px:
                continue
            rows.append((day, asset.upper(), float(px)))
            n += 1
        print(f"{asset}: {n} days", file=sys.stderr)
    rows.sort()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "asset", "price_usd"])
        for d, a, px in rows:
            w.writerow([d, a, f"{px:.10g}"])
    print(f"wrote {len(rows)} rows to {OUT.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
