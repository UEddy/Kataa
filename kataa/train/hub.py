"""Push run artifacts to the Hub as training goes, so a killed job keeps its progress.
Authentication comes from HF_TOKEN in the job's encrypted secrets; it is never logged."""

from __future__ import annotations

import os
from pathlib import Path

RUNS_REPO = os.environ.get("KATAA_RUNS_REPO", "JustEddy1/kataa-runs")


def push_folder(local: str | Path, path_in_repo: str, message: str,
                allow: list[str] | None = None) -> None:
    from huggingface_hub import HfApi

    local = Path(local)
    if not local.exists():
        return
    try:
        HfApi().upload_folder(folder_path=str(local), path_in_repo=path_in_repo,
                              repo_id=RUNS_REPO, repo_type="model", commit_message=message,
                              allow_patterns=allow,
                              ignore_patterns=["*.pt", "optimizer*", "rng_state*", "scheduler*", "README.md",
                                               "**/README.md"])
        print(f"[hub] pushed {local} -> {RUNS_REPO}/{path_in_repo}", flush=True)
    except Exception as e:  # never kill training over an upload
        print(f"[hub] push failed for {path_in_repo}: {type(e).__name__}: {e}", flush=True)


def sync_callback(output_dir: str, path_in_repo: str):
    """transformers TrainerCallback that pushes adapters, logs and trajectories on every save."""
    from transformers import TrainerCallback

    class HubSync(TrainerCallback):
        def on_save(self, args, state, control, **kw):
            push_folder(output_dir, path_in_repo, f"{path_in_repo}: step {state.global_step}")

        def on_train_end(self, args, state, control, **kw):
            push_folder(output_dir, path_in_repo, f"{path_in_repo}: end at step {state.global_step}")

    return HubSync()
