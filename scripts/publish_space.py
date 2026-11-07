"""Create the public Space and upload the replay. Runs in: cloud session ($0, static Space).

    python scripts/publish_space.py --confirm-public

Only with Eddy's explicit OK: this creates a PUBLIC Space (static SDK, free for everyone per the
Spaces docs). The replay data is rebuilt first, so the Space matches docs/results.json. Then run
scripts/check_space_public.py against it before announcing anything.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE = "JustEddy1/kataa-demo"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-public", action="store_true", required=True)
    args = ap.parse_args()
    assert args.confirm_public
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_space_data.py")], check=True)
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(SPACE, repo_type="space", space_sdk="static", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(ROOT / "space"), repo_id=SPACE, repo_type="space",
                      commit_message="Kataa recorded-run replay")
    print(f"[space] https://huggingface.co/spaces/{SPACE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
