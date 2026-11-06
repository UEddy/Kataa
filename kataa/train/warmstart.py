"""Warm start: supervised fine-tuning on the scripted ideal runs. Runs in: HF Job (GPU).

    python -m kataa.train.warmstart --out /work/warmstart --hub-path warmstart [--n-tasks N]
        [--max-steps N] [--epochs E] [--merge-to /work/warmstart-merged]
"""

from __future__ import annotations

import argparse
import json
import time

from kataa.train.common import BASE_MODEL, load_split, precision_kwargs
from kataa.train.hub import push_folder, sync_callback
from kataa.train.sft import build_distill_trainer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hub-path", required=True)
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--merge-to", default=None)
    ap.add_argument("--student", default=BASE_MODEL)
    args = ap.parse_args()

    records = load_split("train", args.n_tasks)
    t0 = time.time()
    dt = build_distill_trainer(
        records, student=args.student, output_dir=args.out,
        num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True, max_length=4096, logging_steps=1,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        report_to="none", lr_scheduler_type="cosine", warmup_steps=5,
        **precision_kwargs(),
    )
    # Attach the Hub sync callback once the factory has built the SFTTrainer.
    factory = dt.trainer_factory
    orig = factory.__call__

    def with_sync(**kw):
        tr = orig(**kw)
        tr.add_callback(sync_callback(args.out, args.hub_path))
        return tr

    dt.trainer_factory = with_sync
    dt.train()
    tr = factory.sft_trainer
    tr.save_model(args.out)
    with open(f"{args.out}/log_history.json", "w") as f:
        json.dump(tr.state.log_history, f, indent=1)
    print(f"[warmstart] done in {time.time() - t0:.0f}s, steps={tr.state.global_step}", flush=True)
    push_folder(args.out, args.hub_path, f"{args.hub_path}: final adapter")

    if args.merge_to:  # GRPO starts from the merged warm-start model with a fresh LoRA
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        pk = precision_kwargs()
        dtype = torch.bfloat16 if pk["bf16"] else torch.float16 if pk["fp16"] else torch.float32
        base = AutoModelForCausalLM.from_pretrained(args.student, dtype=dtype)
        merged = PeftModel.from_pretrained(base, args.out).merge_and_unload()
        merged.save_pretrained(args.merge_to)
        AutoTokenizer.from_pretrained(args.student).save_pretrained(args.merge_to)
        print(f"[warmstart] merged model -> {args.merge_to}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
