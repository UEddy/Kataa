"""Logged-out check of the public Space. Runs in: cloud session ($0).

    python scripts/check_space_public.py --space JustEddy1/kataa-demo [--shots docs/space-shots]

Strict (any failure -> exit 1):
  1. Every file in the Space repo downloads with no credentials (no token, no cookies).
  2. Every huggingface.co / *.hf.space link found in those files returns HTTP 200 with no
     credentials, after redirects. A private or gated repo fails here (401/403/404).
  3. The Space page, opened in a fresh Chromium context (no cookies, no storage), loads with no
     console errors and no failed requests; every request during page load goes to the Space's
     own host (the replay data is bundled; agreed plan). Each control marked `data-check` is
     clicked and must leave at least one `[data-testid=episode]` on the page. Desktop and
     phone-width screenshots are saved for Eddy.

Not checked (other hosts: Lexsi, Cohere, arXiv, Coin Metrics, GitHub, ...): this cloud
session's sandbox may not reach them, so a failure would say nothing about the link. They are
printed as a "verify manually" list, to open in a private tab on a phone before submission.

Credentials are never used: requests go through urllib with no Authorization header, the
huggingface_hub client is never constructed, and every *TOKEN* variable is removed from the
process (and from the browser subprocess) before any request.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

URL_RE = re.compile(r"""https?://[^\s"'<>()\[\]{}`\\]+""")
TEXT_EXT = {".html", ".htm", ".js", ".mjs", ".css", ".json", ".md", ".txt", ".svg", ".yml", ".yaml"}
UA = "kataa-space-check/1.0"


def is_hf(url: str) -> bool:
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    return host in ("huggingface.co", "hf.co") or host.endswith(".huggingface.co") \
        or host.endswith(".hf.space")


def extract_links(text: str) -> set[str]:
    out = set()
    for u in URL_RE.findall(text):
        u = u.rstrip(".,;:!?")
        if "$" in u or "{{" in u:  # template literal (the regex stops at "{"), not a real link
            continue
        if u.startswith(("http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink")):  # XML namespaces
            continue
        out.add(u)
    return out


def classify(links: set[str]) -> tuple[list[str], list[str]]:
    """(strict huggingface links, manual-check links), both sorted."""
    hf = sorted(u for u in links if is_hf(u))
    manual = sorted(u for u in links if not is_hf(u))
    return hf, manual


def fetch(url: str, method: str = "GET", timeout: int = 30) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method=method, headers={"User-Agent": UA})
    assert "Authorization" not in req.headers
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read() if method == "GET" else b""
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:  # network error
        return -1, str(e).encode()


def space_meta(space: str) -> dict:
    st, body = fetch(f"https://huggingface.co/api/spaces/{space}")
    if st != 200:
        raise SystemExit(f"FAIL: Space metadata not public ({st}): {space}")
    return json.loads(body)


def space_host(meta: dict) -> str:
    """The host the Space is served from, as the Hub reports it (a static Space is served from
    <subdomain>.static.hf.space, not <subdomain>.hf.space)."""
    host = urllib.parse.urlsplit(meta.get("host") or "").hostname
    if not host:
        raise SystemExit("FAIL: the Hub did not report a serving host for the Space")
    return host.lower()


def browser_check(url: str, host: str, shots: Path | None) -> list[str]:
    """Fresh, logged-out Chromium via the session's pre-installed Node Playwright
    (scripts/space_browser_check.mjs); returns the problems it found."""
    import subprocess

    js = Path(__file__).with_name("space_browser_check.mjs")
    cmd = ["node", str(js), url, host, str(shots) if shots else ""]
    env = {k: v for k, v in os.environ.items() if "TOKEN" not in k.upper()}
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])["problems"]
    except Exception:
        return [f"browser check crashed: {r.stderr.strip()[-500:]}"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="owner/name")
    ap.add_argument("--shots", default=None, help="folder for screenshots")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    for k in [k for k in os.environ if "TOKEN" in k.upper()]:
        os.environ.pop(k)  # nothing below can authenticate

    fails: list[str] = []
    links: set[str] = set()
    meta = space_meta(args.space)
    files = [x["rfilename"] for x in meta.get("siblings", [])]
    for f in files:
        url = f"https://huggingface.co/spaces/{args.space}/resolve/main/{urllib.parse.quote(f)}"
        st, body = fetch(url)
        if st != 200:
            fails.append(f"Space file {f}: HTTP {st}")
        elif Path(f).suffix.lower() in TEXT_EXT:
            links |= extract_links(body.decode("utf-8", "replace"))
    hf, manual = classify(links)
    for u in hf:
        st, _ = fetch(u)
        if st != 200:
            fails.append(f"Hub link {u}: HTTP {st}")
    host = space_host(meta)
    if not args.no_browser:
        fails += browser_check(f"https://{host}/", host, Path(args.shots) if args.shots else None)

    print(f"Space files checked: {len(files)}; huggingface links checked: {len(hf)}")
    print("RESULT:", "PASS" if not fails else f"FAIL ({len(fails)})")
    for x in fails:
        print("  FAIL", x)
    print(f"\nVerify manually ({len(manual)} links, not reachable from the sandbox or not checked):")
    for u in manual:
        print("  ", u)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
