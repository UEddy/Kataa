"""Evaluate one full model with vLLM. Runs in: HF Job (l40sx1).

    python -m kataa.eval.vllm_eval --name ws150 --model /work/merged/ws150 --split val \
        --n-tasks 40 --samples 4 --temperature 0.9 --out /work/eval --hub-path eval/<name>

Checkpoints are merged into full models first (kataa.train.merge): vLLM 0.22.1's LoRA path
crashes on Tiny Aya ("'VocabParallelEmbeddingWithLoRA' object has no attribute
'quant_method'", job 6aedb555c78863f9c63409ca). Run one model per process so GPU memory is
fully released between models; results merge into <out>/summary.json.
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
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default=BASE_MODEL, help="hub id or merged model dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hub-path", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    llm = LLM(model=args.model, tokenizer=BASE_MODEL, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.85, seed=0)
    sp = SamplingParams(temperature=args.temperature, top_p=1.0, max_tokens=args.max_tokens,
                        seed=0 if args.temperature == 0 else None)
    records = load_split(args.split, args.n_tasks)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    name = args.name

    def render(conv):
        return tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

    def generate(convs):
        res = llm.generate([render(c) for c in convs], sp, use_tqdm=False)
        return [r.outputs[0].text for r in res]

    def count_tokens(conv):
        return len(tok(render(conv), add_special_tokens=False)["input_ids"])

    t0 = time.time()
    eps = run_episodes(generate, records, n_samples=args.samples,
                       max_prompt_tokens=8192 - args.max_tokens, count_tokens=count_tokens)
    rows = score(eps)
    summary[name] = {**summarize(rows), "model": args.model, "split": args.split,
                     "samples": args.samples, "temperature": args.temperature,
                     "seconds": round(time.time() - t0, 1)}
    print(f"[eval] {name}: {json.dumps(summary[name])}", flush=True)
    with open(out / f"{name}.rows.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(out / f"{name}.trajectories.jsonl", "w") as f:
        f.writelines(json.dumps(to_agenttune_jsonl(e)) + "\n" for e in eps)
    summary_path.write_text(json.dumps(summary, indent=1))
    push_folder(out, args.hub_path, f"{args.hub_path}: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
