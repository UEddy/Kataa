"""Merge LoRA adapters into Tiny Aya, in order, and save the full model.
Runs in: HF Job.   python -m kataa.train.merge --adapter WARMSTART [--adapter GRPO_RUN] --out DIR

GRPO runs train a fresh LoRA on top of the merged warm-start model, so a run's adapter must be
applied after the warm-start adapter it started from (an eval that merged run 1 onto base Tiny
Aya alone silently dropped the warm start: eval/run1/8c050d7f4788)."""

from __future__ import annotations

import argparse

from kataa.train.common import BASE_MODEL, precision_kwargs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", action="append", required=True, help="repeat, in order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE_MODEL)
    args = ap.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pk = precision_kwargs()
    dtype = torch.bfloat16 if pk["bf16"] else torch.float16 if pk["fp16"] else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype)
    for adapter in args.adapter:
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print(f"[merge] {args.adapter} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
