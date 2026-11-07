"""DPO repair: pair building and a CPU run of AgentTune's DPO path. Runs in: cloud session (CPU)."""

import json
from pathlib import Path

import pytest

pytest.importorskip("agenttune")

from kataa.env.world import TaskSpec  # noqa: E402
from kataa.eval.runner import run_episodes, score, to_agenttune_jsonl  # noqa: E402
from kataa.policy.ideal import DECISION_TOOL, call_list, ideal_turns  # noqa: E402
from kataa.rewards.episode import parse_calls_from_text  # noqa: E402
from kataa.rewards.rewards import is_unsafe  # noqa: E402
from kataa.train.common import load_split  # noqa: E402
from kataa.train.dpo import (build_examples, decision_turn_index,  # noqa: E402
                             ideal_decision_turn)

WRONG = {"margin_call": "no_action", "no_action": "margin_call", "escalate": "margin_call"}


def train_records(n_per_condition=3):
    picked, seen = [], {}
    for r in load_split("train"):
        if seen.get(r["condition"], 0) < n_per_condition:
            seen[r["condition"]] = seen.get(r["condition"], 0) + 1
            picked.append(r)
    return picked


def unsafe_policy(records):
    """Ideal evidence turns, then the wrong decision (unsafe for every answer type), then a
    second, correct decision afterwards (only the first decision counts)."""
    scripts = {}
    for r in records:
        spec = TaskSpec.from_dict(json.loads(r["spec"]))
        turns = ideal_turns(spec)
        k = next(i for i, t in enumerate(turns)
                 if any(c[0] in DECISION_TOOL.values() for c in parse_calls_from_text(t)[0] or []))
        bad = call_list([(DECISION_TOOL[WRONG[r["answer"]]],
                          {"loan_id": r["loan_id"], "reason": "manager said so"})])
        scripts[r["loan_id"]] = turns[:k] + ["Acting now.\n" + bad] + turns[k:]
    return scripts


def run(records, scripts):
    def generate(convs):
        out = []
        for c in convs:
            user = next(m["content"] for m in c if m["role"] == "user")
            loan = next(k for k in scripts if k in user)
            i = sum(m["role"] == "assistant" for m in c)
            out.append(scripts[loan][i] if i < len(scripts[loan]) else "Done.")
        return out
    eps = run_episodes(generate, records)
    return score(eps), [to_agenttune_jsonl(e) for e in eps]


def test_pairs_cut_at_the_decision():
    recs = train_records()
    rows, trajs = run(recs, unsafe_policy(recs))
    truth = {r["task_id"]: r for r in recs}
    # Missing a due margin call is wrong but not unsafe (headline definition), unless an attack
    # pushed for it; every other scripted wrong decision is unsafe.
    expect = [r for r in rows if is_unsafe(WRONG[r["answer"]], r["answer"],
                                           truth[r["task_id"]]["attack_target"])]
    assert [r["task_id"] for r in rows if r["unsafe"]] == [r["task_id"] for r in expect]
    assert len(expect) >= 15
    examples, report = build_examples(load_split("train"), rows, trajs)
    assert report["pairs"] == len(expect) and report["episodes"] == len(recs)
    by_tid = {t["metadata"]["task_id"]: t["metadata"]["conversation"] for t in trajs}
    for ex in examples:
        tid = ex.trajectory_id.split(":")[0]
        conv = by_tid[tid]
        k = decision_turn_index(conv)
        # prompt = the model's own run up to its decision; no tool output in either completion
        assert [m["content"] for m in ex.prompt] == [m["content"] for m in conv[:k]]
        assert ex.prompt[-1]["role"] == "user" and ex.prompt[0]["role"] == "system"
        assert len(ex.chosen) == 1 and len(ex.rejected) == 1
        assert ex.rejected[0]["content"] == conv[k]["content"]
        spec = TaskSpec.from_dict(json.loads(truth[tid]["spec"]))
        assert ex.chosen[0]["content"] == ideal_decision_turn(spec)
        chosen_call = parse_calls_from_text(ex.chosen[0]["content"])[0][0][0]
        assert chosen_call == DECISION_TOOL[truth[tid]["answer"]]


def test_safe_episodes_give_no_pairs():
    recs = train_records(1)
    ideal = {r["loan_id"]: ideal_turns(TaskSpec.from_dict(json.loads(r["spec"]))) for r in recs}
    rows, trajs = run(recs, ideal)
    examples, report = build_examples(load_split("train"), rows, trajs)
    assert examples == [] and report["pairs"] == 0


def test_rejects_non_train_tasks():
    recs = [load_split("heldout")[0], load_split("test")[0], load_split("val")[0]]
    for r in recs:
        rows, trajs = run([r], unsafe_policy([r]))
        with pytest.raises(ValueError, match="non-train"):
            build_examples(load_split("train") + [r], rows, trajs)


def test_agenttune_build_dataset_rows():
    from agenttune.agentic import build_dataset
    recs = train_records(1)
    rows, trajs = run(recs, unsafe_policy(recs))
    examples, _ = build_examples(load_split("train"), rows, trajs)
    ds = build_dataset(examples)
    assert len(ds) == len(examples) > 0
    assert set(ds[0]) == {"prompt", "chosen", "rejected"}


def test_cpu_dpo_through_agenttune(tmp_path):
    """Tiny random Cohere2 model + Tiny Aya's tokenizer, real pairs (shortened to fit CPU memory),
    the job's exact create_agentic_trainer arguments. Checks the loss runs, both loss terms are
    accepted, LoRA targets resolve on Cohere2 and an adapter is saved."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from agenttune.agentic import build_dataset
    from agenttune.core.backend_factory import create_agentic_trainer
    from kataa.train.common import BASE_MODEL
    from kataa.train.dpo import filter_by_length
    from kataa.train.sft import LORA

    recs = [r for r in train_records(1) if r["answer"] != "margin_call"][:4]
    rows, trajs = run(recs, unsafe_policy(recs))
    examples, _ = build_examples(load_split("train"), rows, trajs)
    pairs = build_dataset(examples)
    for p in pairs:  # keep the shape (system, ..., last tool result), cut the long agreement text
        p["prompt"] = [p["prompt"][0] | {"content": p["prompt"][0]["content"][:300]},
                       p["prompt"][-1] | {"content": p["prompt"][-1]["content"][:300]}]
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    pairs, dropped = filter_by_length(pairs, tok, max_tokens=2048)
    assert pairs and dropped == 0
    cfg = AutoConfig.from_pretrained(BASE_MODEL)
    cfg.update(dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                    num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                    layer_types=["sliding_attention", "full_attention"]))
    torch.manual_seed(0)
    lm = AutoModelForCausalLM.from_config(cfg)
    out = tmp_path / "dpo"
    trainer = create_agentic_trainer(
        "dpo", backend="trl", model=lm, processing_class=tok, train_dataset=Dataset.from_list(pairs),
        peft_config=LoraConfig(**LORA), output_dir=str(out), beta=0.1, learning_rate=2e-5,
        num_train_epochs=1, max_steps=2, per_device_train_batch_size=1, gradient_accumulation_steps=2,
        loss_type=["sigmoid", "sft"], loss_weights=[1.0, 0.2], max_length=None,
        gradient_checkpointing=False, warmup_steps=0, lr_scheduler_type="cosine", logging_steps=1,
        save_steps=20, save_total_limit=2, report_to=[], seed=0, bf16=False, fp16=False, use_cpu=True)
    trainer.train()
    hist = trainer.trainer.state.log_history
    assert any("rewards/accuracies" in h for h in hist)
    assert (out / "adapter_model.safetensors").exists()
    cfgj = json.loads((out / "adapter_config.json").read_text())
    assert cfgj["r"] == 16 and cfgj["lora_alpha"] == 32
    assert Path(out / "training_stats.json").exists()


def test_pair_tokenization_matches_trl():
    """Prompt tokens are a prefix of prompt+completion (TRL warns and mis-splits otherwise), each
    completion ends at the end-of-turn token, and the length filter counts real tokens."""
    from transformers import AutoTokenizer

    from agenttune.agentic import build_dataset
    from kataa.train.common import BASE_MODEL
    from kataa.train.dpo import chat_ids, filter_by_length

    recs = [r for r in train_records(1) if r["answer"] != "margin_call"]
    rows, trajs = run(recs, unsafe_policy(recs))
    pairs = build_dataset(build_examples(load_split("train"), rows, trajs)[0])
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    for p in pairs:
        pi = chat_ids(tok, p["prompt"], add_generation_prompt=True)
        for c in ("chosen", "rejected"):
            full = chat_ids(tok, p["prompt"] + p[c])
            assert full[: len(pi)] == pi and len(full) > len(pi) + 10
            assert tok.decode(full[len(pi):]).endswith("<|END_OF_TURN_TOKEN|>")
    kept, dropped = filter_by_length(pairs, tok, max_tokens=10)
    assert kept == [] and dropped == len(pairs)
    kept, dropped = filter_by_length(pairs, tok)
    assert dropped == 0 and len(kept) == len(pairs)


def test_completion_only_logits_matches_trl():
    """The memory fix (logits only from the first completion position) gives the same loss and
    metrics as TRL's unpatched _compute_loss, for policy and reference passes, with the SFT term."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.models.cohere2.modeling_cohere2 import Cohere2ForCausalLM
    from trl import DPOTrainer

    from agenttune.agentic import build_dataset
    from agenttune.core.backend_factory import create_agentic_trainer
    from kataa.train.common import BASE_MODEL
    from kataa.train.dpo import install_completion_only_logits
    from kataa.train.sft import LORA

    recs = [r for r in train_records(1) if r["answer"] != "margin_call"][:2]
    rows, trajs = run(recs, unsafe_policy(recs))
    pairs = build_dataset(build_examples(load_split("train"), rows, trajs)[0])
    for p in pairs:  # shorter prompts so the unpatched pass fits CPU memory
        p["prompt"] = [p["prompt"][0] | {"content": p["prompt"][0]["content"][:400]},
                       p["prompt"][-1] | {"content": p["prompt"][-1]["content"][:400]}]
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    cfg = AutoConfig.from_pretrained(BASE_MODEL)
    cfg.update(dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                    num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                    layer_types=["sliding_attention", "full_attention"]))
    torch.manual_seed(0)
    lm = AutoModelForCausalLM.from_config(cfg)
    wrapper = create_agentic_trainer(
        "dpo", backend="trl", model=lm, processing_class=tok, train_dataset=Dataset.from_list(pairs),
        peft_config=LoraConfig(**LORA), output_dir="/tmp/kataa-dpo-eq", beta=0.1,
        loss_type=["sigmoid", "sft"], loss_weights=[1.0, 0.2], max_length=None,
        per_device_train_batch_size=1, report_to=[], bf16=False, fp16=False, use_cpu=True)
    wrapper.setup_data()
    wrapper.setup_trainer()
    tr = wrapper.trainer
    # make the LoRA non-zero so policy != reference and the loss is not trivially log(2)
    with torch.no_grad():
        for n, prm in tr.model.named_parameters():
            if "lora_B" in n:
                prm.normal_(0, 0.05)
    batch = next(iter(tr.get_train_dataloader()))
    orig_loss, orig_fwd = DPOTrainer._compute_loss, Cohere2ForCausalLM.forward
    try:
        tr._metrics = {"train": __import__("collections").defaultdict(list)}
        a = orig_loss(tr, tr.model, batch, False).item()
        ma = {k: list(v) for k, v in tr._metrics["train"].items()}
        install_completion_only_logits()
        patched = Cohere2ForCausalLM.forward
        seen = []

        def spy(self, *args, **kw):
            out = patched(self, *args, **kw)
            seen.append((out.logits.shape[1], batch["input_ids"].shape[1]))
            return out
        Cohere2ForCausalLM.forward = spy
        tr._metrics = {"train": __import__("collections").defaultdict(list)}
        b = tr._compute_loss(tr.model, batch, False).item()
        mb = {k: list(v) for k, v in tr._metrics["train"].items()}
    finally:
        DPOTrainer._compute_loss, Cohere2ForCausalLM.forward = orig_loss, orig_fwd
        for cls in (DPOTrainer, Cohere2ForCausalLM):
            if "_kataa_completion_only" in cls.__dict__:
                delattr(cls, "_kataa_completion_only")
    assert abs(a - b) < 1e-4 * max(1, abs(a)), (a, b)
    assert abs(a - 0.6931) > 1e-3, "policy == reference: the comparison would be trivial"
    assert any(k.startswith("rewards/") for k in ma) and set(ma) == set(mb)
    assert seen and all(n < full for n, full in seen), seen  # patched pass really sliced
    for k in ma:
        if "logits" in k:  # mean logit over completion tokens: same tokens, same values
            assert ma[k] == pytest.approx(mb[k], rel=1e-4, abs=1e-5), k
        if k.startswith("rewards/") or k.startswith("logps/"):
            assert ma[k] == pytest.approx(mb[k], rel=1e-4, abs=1e-5), k
