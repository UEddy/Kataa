"""Merge a LoRA adapter into Tiny Aya and save the full model (the GRPO starting point).
Runs in: HF Job.   python -m kataa.train.merge --adapter DIR --out DIR"""

from __future__ import annotations

import argparse

from kataa.train.common import BASE_MODEL, precision_kwargs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE_MODEL)
    args = ap.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pk = precision_kwargs()
    dtype = torch.bfloat16 if pk["bf16"] else torch.float16 if pk["fp16"] else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype)
    PeftModel.from_pretrained(base, args.adapter).merge_and_unload().save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print(f"[merge] {args.adapter} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
