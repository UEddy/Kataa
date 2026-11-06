"""GRPO on the lending desk with AgentTune. Runs in: HF Job (GPU).

    python -m kataa.train.grpo --model /work/warmstart-merged --reward naive|hardened \
        --out /work/run1 --hub-path run1 [--n-tasks N] [--max-steps N]
"""

from __future__ import annotations

import argparse
import json
import time

from kataa.env import registry, tools
from kataa.protocol import MAX_TOOL_TURNS, TOOL_RESULT_FORMAT, TOOLS_PROMPT
from kataa.rewards import hardened_reward, naive_reward
from kataa.train.common import load_split, peak_mem, precision_kwargs
from kataa.train.hub import push_folder, sync_callback

REWARDS = {"naive": naive_reward, "hardened": hardened_reward}
KEEP = ("prompt", "answer", "spec", "attack_target", "task_id", "condition", "attack_family")


def vllm_kwargs(args) -> dict:
    if not args.vllm:
        return {"use_vllm": False}
    # TRL's vLLM importance-sampling correction (default on, sequence_mask, cap 3.0) zeroes the
    # loss for AgentTune's multi-turn rollouts: the trainer scores the model's turns as one
    # concatenated sequence (tool results as raw text), while vLLM sampled each turn from the
    # fully rendered chat. The summed log-prob gap makes the per-sequence ratio collapse, and
    # run-1 attempt 6aedc601 trained with grad_norm ~1e-16. Off = same objective as the
    # transformers-generation path (where the first-step loss was 0.248).
    return {"use_vllm": True, "vllm_mode": "colocate",
            "vllm_gpu_memory_utilization": args.vllm_mem, "vllm_max_model_length": 8192,
            "vllm_importance_sampling_correction": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--reward", choices=REWARDS, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hub-path", required=True)
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-completion-length", type=int, default=768)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--save-steps", type=int, default=20)
    ap.add_argument("--vllm", action="store_true", help="colocated vLLM generation")
    ap.add_argument("--vllm-mem", type=float, default=0.35)
    args = ap.parse_args()

    from datasets import Dataset

    from agenttune.core.backend_factory import create_agentic_trainer

    records = load_split("train", args.n_tasks)
    registry.load_dir()  # tools serve every split's loans by ID
    ds = Dataset.from_list([{k: r[k] for k in KEEP} for r in records])

    t0 = time.time()
    trainer = create_agentic_trainer(
        algorithm="grpo", backend="trl",
        model=args.model, train_dataset=ds, tools=tools.TOOLS,
        reward_funcs=[REWARDS[args.reward]],
        output_dir=args.out, max_steps=args.max_steps, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, num_generations=args.num_generations,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        max_completion_length=args.max_completion_length, beta=args.beta,
        temperature=args.temperature, warmup_steps=0, logging_steps=1,
        save_steps=args.save_steps, save_total_limit=2, gradient_checkpointing=True,
        max_steps_per_turn=MAX_TOOL_TURNS, tools_fallback_prompt=TOOLS_PROMPT,
        tool_result_format=TOOL_RESULT_FORMAT, report_to="none",
        log_completions=False,
        peft_config=dict(r=16, lora_alpha=32, target_modules="all-linear",
                         task_type="CAUSAL_LM"),
        callbacks=[sync_callback(args.out, args.hub_path)],
        **vllm_kwargs(args),
        **precision_kwargs(),
    )
    trainer.train()
    hist = trainer.trainer.state.log_history
    with open(f"{args.out}/log_history.json", "w") as f:
        json.dump(hist, f, indent=1)
    print(f"[grpo] {args.reward}: done in {time.time() - t0:.0f}s, "
          f"steps={trainer.trainer.state.global_step}, {peak_mem()}", flush=True)
    push_folder(args.out, args.hub_path, f"{args.hub_path}: final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
