"""Targeted DPO repair of run 2 through AgentTune's self-heal/DPO path.
Runs in: HF Job (l40sx1) for training; pair building is CPU-only and unit-tested.

    python -m kataa.train.dpo --model /work/merged/run2 --rows R.rows.jsonl \
        --trajectories R.trajectories.jsonl --out /work/dpo --hub-path dpo/<sha>

Agreed with Eddy before the run (2026-11-07):
- Failures come from run 2 itself, greedy, on the **train split only** (no test, held-out or val
  task can enter). A failure is an unsafe decision by `kataa.rewards.rewards.is_unsafe`, the
  headline definition, checked against the environment's ground truth.
- Pairs are cut at the decision (option B). The prompt is run 2's own conversation up to the turn
  in which it first recorded a decision; `rejected` is that turn; `chosen` is the scripted ideal
  decision turn (correct decision tool, notes and reason from kataa.policy.ideal). The loss
  therefore covers only the decision turn, never tool output, and the prefix is on-policy.
- Each pair becomes AgentTune's `TrainingExample` and goes through
  `agenttune.agentic.build_dataset`, the self-heal loop's dataset step, then
  `create_agentic_trainer("dpo", ...)` (TRL DPOTrainer, offline mode: the dataset has real
  chosen/rejected columns). AgentTune's LLM-based FailureDetector/Classifier are not used: the
  environment's ground truth is a stronger detector, and AuditReader.extract_dpo_pairs is listed
  as broken in AgentTune's Known Issues.
- Loss: sigmoid DPO plus a small SFT term on the chosen decision turns
  (loss_type=["sigmoid", "sft"], loss_weights=[1.0, 0.2]) against escalating everything.
- New LoRA (r=16, alpha 32, all-linear) on the merged WS2+run2 model; the reference policy is
  the same model with the adapter disabled.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from kataa.env import registry
from kataa.env.world import DECISION_TOOLS, TaskSpec, decide
from kataa.policy.ideal import ideal_turns
from kataa.rewards.episode import parse_calls_from_text
from kataa.rewards.rewards import is_unsafe

MIN_PAIRS = 100        # below this the job stops before training (agreed)
MAX_PAIR_TOKENS = 7680  # prompt + longer completion; Tiny Aya's context is 8,192
TRAIN_PREFIX = "train-"


def decision_turn_index(conv: list[dict]) -> int | None:
    """Index of the first assistant message whose call list contains a decision tool."""
    for i, m in enumerate(conv):
        if m.get("role") != "assistant":
            continue
        calls, _ = parse_calls_from_text(m.get("content") or "")
        if calls and any(name in DECISION_TOOLS for name, _args, _ok in calls):
            return i
    return None


def first_decision_name(turn: dict) -> str | None:
    calls, _ = parse_calls_from_text(turn.get("content") or "")
    for name, _args, _ok in calls or []:
        if name in DECISION_TOOLS:
            return name
    return None


def ideal_decision_turn(spec: TaskSpec) -> str:
    """The scripted ideal run's decision turn (notes + the correct decision call)."""
    turns = ideal_turns(spec)
    for t in turns:
        calls, _ = parse_calls_from_text(t)
        if calls and any(name in DECISION_TOOLS for name, _a, _o in calls):
            return t
    raise ValueError(f"ideal run for {spec.loan_id} has no decision turn")


DECISION_OF_TOOL = DECISION_TOOLS  # tool name -> decision


def build_examples(records: list[dict], rows: list[dict], trajectories: list[dict]):
    """Unsafe run-2 episodes on train tasks -> AgentTune TrainingExamples (option B pairs).

    Returns (examples, report). Raises if any non-train task appears in the inputs.
    """
    from agenttune.decide.closed_loop.contracts import TrainingExample

    by_id = {r["task_id"]: r for r in records}
    convs = {(t["metadata"]["task_id"], t["metadata"].get("sample", 0)): t["metadata"]["conversation"]
             for t in trajectories}
    report: Counter = Counter()
    examples = []
    for row in rows:
        tid = row["task_id"]
        if not tid.startswith(TRAIN_PREFIX) or tid not in by_id:
            raise ValueError(f"non-train task in DPO inputs: {tid}")
        report["episodes"] += 1
        if not row["unsafe"]:
            continue
        report["unsafe"] += 1
        rec = by_id[tid]
        spec = TaskSpec.from_dict(json.loads(rec["spec"]))
        registry.register(spec)
        answer = decide(spec)[0]
        assert answer == rec["answer"], f"{tid}: ground truth drifted from the task file"
        conv = convs.get((tid, row.get("sample", 0)))
        k = decision_turn_index(conv) if conv else None
        if k is None:
            report["skipped_no_decision_turn"] += 1
            continue
        rejected_turn = {"role": "assistant", "content": conv[k]["content"]}
        rej = DECISION_OF_TOOL[first_decision_name(rejected_turn)]
        # Sanity: the turn we reject is the unsafe decision the row was scored on.
        if not is_unsafe(rej, answer, rec["attack_target"]):
            report["skipped_turn_not_unsafe"] += 1
            continue
        chosen_text = ideal_decision_turn(spec)
        assert DECISION_OF_TOOL[first_decision_name({"content": chosen_text})] == answer
        prompt = [{"role": m["role"], "content": m["content"]} for m in conv[:k]]
        examples.append(TrainingExample(
            trajectory_id=f"{tid}:s{row.get('sample', 0)}",
            original_failure_type="unsafe_decision",
            root_cause=f"{rec['condition']}: decided {rej}, correct is {answer}"
                       + (f" (attack {rec['attack_family']})" if rec["attack_family"] else ""),
            prompt=prompt,
            chosen=[{"role": "assistant", "content": chosen_text}],
            rejected=[rejected_turn],
        ))
        report[f"pair_condition:{rec['condition']}"] += 1
        report[f"pair_chosen:{answer}"] += 1
        report[f"pair_rejected:{rej}"] += 1
    report["pairs"] = len(examples)
    return examples, dict(sorted(report.items()))


def chat_ids(tok, messages: list[dict], **kw) -> list[int]:
    """Token ids exactly as TRL's DPOTrainer._tokenize builds them. (transformers 5 returns a
    BatchEncoding from apply_chat_template(tokenize=True), so len() of it is the key count.)"""
    return tok.apply_chat_template(messages, tokenize=True, return_dict=True, **kw)["input_ids"]


def filter_by_length(rows: list[dict], tok, max_tokens: int = MAX_PAIR_TOKENS):
    """Drop pairs that would not fit (TRL's keep_start truncation would cut the completion)."""
    kept, dropped = [], 0
    for r in rows:
        n = max(len(chat_ids(tok, r["prompt"] + r[c])) for c in ("chosen", "rejected"))
        if n <= max_tokens:
            kept.append(r)
        else:
            dropped += 1
    return kept, dropped


_FULL_IDS = "kataa_full_input_ids"


def install_completion_only_logits(trainer_cls=None, model_cls=None) -> None:
    """Compute logits only from the position before the first completion token (job 6aef03f2 ran
    out of memory without this).

    TRL 1.7.1's DPOTrainer._compute_loss takes full-length logits from the policy and the
    reference pass, then masks everything but the completion. With ~4.5K-token prompts and Tiny
    Aya's 262K vocabulary that is ~4.7 GB of bf16 logits per pass, plus their fp32 log-softmax
    and gradients. Here, before the original _compute_loss runs:
      - the full input_ids go to the model under a side key, with logits_to_keep=M, so the model
        still attends over the whole prompt but projects only the last M positions to the
        vocabulary (M = L - c + 1, c = first completion index in the batch);
      - TRL's loss code gets input_ids[:, c-1:] and completion_mask[:, c-1:], which line up with
        those M logits exactly as the full tensors would (shift by one, masked to completion).
    The loss, the SFT term and the logged metrics are unchanged (tests/test_dpo.py checks this
    against the unpatched trainer). attention_mask is left full-length for the model.
    """
    from trl import DPOTrainer

    trainer_cls = trainer_cls or DPOTrainer
    if getattr(trainer_cls, "_kataa_completion_only", False):
        return
    orig_loss = trainer_cls._compute_loss

    def _compute_loss(self, model, inputs, return_outputs):
        ids, cm = inputs["input_ids"], inputs["completion_mask"]
        has = cm.bool().any(dim=1)
        if not bool(has.all()):
            return orig_loss(self, model, inputs, return_outputs)
        c = int(cm.float().argmax(dim=1).min())  # first completion index across the batch
        if c < 1:
            return orig_loss(self, model, inputs, return_outputs)
        inputs = dict(inputs)
        inputs[_FULL_IDS] = ids
        inputs["input_ids"] = ids[:, c - 1:]
        inputs["completion_mask"] = cm[:, c - 1:]
        return orig_loss(self, model, inputs, return_outputs)

    trainer_cls._compute_loss = _compute_loss
    trainer_cls._kataa_completion_only = True

    def wrap_forward(cls):
        if getattr(cls, "_kataa_completion_only", False):
            return
        orig_fwd = cls.forward

        def forward(self, *args, **kwargs):
            full = kwargs.pop(_FULL_IDS, None)
            if full is not None:
                kept = kwargs["input_ids"].shape[1]
                kwargs["input_ids"] = full
                kwargs["logits_to_keep"] = kept
                out = orig_fwd(self, *args, **kwargs)
                assert out.logits.shape[1] == kept, (out.logits.shape, kept)
                return out
            return orig_fwd(self, *args, **kwargs)

        cls.forward = forward
        cls._kataa_completion_only = True

    if model_cls is None:
        from transformers.models.cohere2.modeling_cohere2 import Cohere2ForCausalLM as model_cls
    wrap_forward(model_cls)


def read_jsonl(p) -> list[dict]:
    return [json.loads(x) for x in Path(p).open()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged WS2+run2 model dir")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hub-path", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--sft-weight", type=float, default=0.2)
    ap.add_argument("--dry-run", action="store_true", help="build and report pairs, no training")
    args = ap.parse_args()

    from agenttune.agentic import build_dataset
    from kataa.train.common import BASE_MODEL, load_split
    from kataa.train.hub import push_folder

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    examples, report = build_examples(load_split("train"), read_jsonl(args.rows),
                                      read_jsonl(args.trajectories))
    rows = build_dataset(examples)  # AgentTune self-heal loop: TrainingExample -> preference row

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows, too_long = filter_by_length(rows, tok)
    report.update({"dropped_too_long": too_long, "pairs_trained": len(rows)})
    print(f"[dpo] pairs: {json.dumps(report)}", flush=True)
    with open(out / "pairs.jsonl", "w") as f:
        for ex, r in zip(examples, rows):
            f.write(json.dumps({"trajectory_id": ex.trajectory_id, "root_cause": ex.root_cause,
                                **r}) + "\n")
    (out / "pairs_report.json").write_text(json.dumps(report, indent=1))
    if not args.dry_run:
        push_folder(out, args.hub_path, f"{args.hub_path}: pairs", allow=["pairs*"])
    if len(rows) < MIN_PAIRS:
        print(f"[dpo] only {len(rows)} pairs (< {MIN_PAIRS}); stopping before training", flush=True)
        return 3
    if args.dry_run:
        return 0

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM

    from agenttune.core.backend_factory import create_agentic_trainer
    from kataa.train.common import peak_mem, precision_kwargs
    from kataa.train.hub import sync_callback
    from kataa.train.sft import LORA

    install_completion_only_logits()
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    trainer = create_agentic_trainer(
        "dpo", backend="trl",
        model=lm, processing_class=tok, train_dataset=Dataset.from_list(rows),
        peft_config=LoraConfig(**LORA),
        output_dir=str(out), beta=args.beta, learning_rate=args.lr, num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum,
        loss_type=["sigmoid", "sft"], loss_weights=[1.0, args.sft_weight],
        max_length=None, gradient_checkpointing=True, warmup_steps=5, lr_scheduler_type="cosine",
        logging_steps=1, save_steps=20, save_total_limit=2, report_to=[], seed=0,
        callbacks=[sync_callback(str(out), args.hub_path)], **precision_kwargs(),
    )
    stats = trainer.train()
    hist = trainer.trainer.state.log_history
    (out / "log_history.json").write_text(json.dumps(hist, indent=1))
    acc = [h["rewards/accuracies"] for h in hist if "rewards/accuracies" in h]
    mar = [h["rewards/margins"] for h in hist if "rewards/margins" in h]
    print(f"[dpo] done: {json.dumps(stats, default=str)[:400]} | reward acc first/last "
          f"{acc[:1]} {acc[-1:]} | margin last {mar[-1:]} | {peak_mem()}", flush=True)
    push_folder(out, args.hub_path, f"{args.hub_path}: final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
