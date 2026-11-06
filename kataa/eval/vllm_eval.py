"""Evaluate base Tiny Aya and LoRA checkpoints with vLLM. Runs in: HF Job (l40sx1).

    python -m kataa.eval.vllm_eval --split val --n-tasks 40 --samples 4 --temperature 0.9 \
        --adapter step25=/work/ck/checkpoint-25 --adapter final=/work/ck/final --out /work/eval \
        --hub-path eval/<name>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kataa.eval.runner import run_episodes, score, summarize, to_agenttune_jsonl
from kataa.train.common import BASE_MODEL, load_split
from kataa.train.hub import push_folder


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--adapter", action="append", default=[], help="name=path (repeatable)")
    ap.add_argument("--include-base", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hub-path", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    llm = LLM(model=BASE_MODEL, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.85, enable_lora=bool(args.adapter), max_lora_rank=16,
              max_loras=1, seed=0)
    sp = SamplingParams(temperature=args.temperature, top_p=1.0, max_tokens=args.max_tokens,
                        seed=0 if args.temperature == 0 else None)
    records = load_split(args.split, args.n_tasks)
    models = ([("base", None)] if args.include_base else []) + \
        [tuple(a.split("=", 1)) for a in args.adapter]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for i, (name, path) in enumerate(models):
        lora = LoRARequest(name, i + 1, path) if path else None

        def generate(convs):
            prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                       for c in convs]
            res = llm.generate(prompts, sp, lora_request=lora, use_tqdm=False)
            return [r.outputs[0].text for r in res]

        def count_tokens(conv):
            return len(tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)) // 3

        t0 = time.time()
        eps = run_episodes(generate, records, n_samples=args.samples,
                           max_prompt_tokens=8192 - args.max_tokens, count_tokens=count_tokens)
        rows = score(eps)
        summary[name] = {**summarize(rows), "seconds": round(time.time() - t0, 1)}
        print(f"[eval] {name}: {json.dumps(summary[name])}", flush=True)
        with open(out / f"{name}.rows.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        with open(out / f"{name}.trajectories.jsonl", "w") as f:
            f.writelines(json.dumps(to_agenttune_jsonl(e)) + "\n" for e in eps)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        push_folder(out, args.hub_path, f"{args.hub_path}: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
