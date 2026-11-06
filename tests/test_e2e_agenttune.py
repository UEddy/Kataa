"""End-to-end CPU test of the agent loop: AgentTune's real tool-calling rollout
(create_rollout_fn -> _execute_trajectory) driven by a scripted policy instead of a model, with
our real tools and rewards. Runs in: cloud session (CPU). Uses Tiny Aya's tokenizer when the HF
token can read it, else SmolLM2's (ungated, and its template also drops `tools`)."""

import json
import warnings
from pathlib import Path

import pytest

pytest.importorskip("agenttune")

from agenttune.agentic.rollout_engines import rollout_factory as rf  # noqa: E402

from kataa.env import registry, tools  # noqa: E402
from kataa.env.world import TaskSpec  # noqa: E402
from kataa.policy.ideal import call_list, ideal_turns  # noqa: E402
from kataa.protocol import MAX_TOOL_TURNS, TOOL_RESULT_FORMAT, TOOLS_PROMPT  # noqa: E402
from kataa.rewards import hardened_reward, naive_reward  # noqa: E402
from kataa.rewards.rewards import R_CORRECT, R_FORMAT  # noqa: E402
from kataa.train.sft import ideal_conversation, tool_schemas, turn_rows  # noqa: E402

TASKS = Path(__file__).resolve().parents[1] / "data" / "tasks"
TOKENIZERS = ["CohereLabs/tiny-aya-global", "HuggingFaceTB/SmolLM2-135M-Instruct"]


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    for name in TOKENIZERS:
        try:
            t = AutoTokenizer.from_pretrained(name)
            print(f"\n[e2e] tokenizer: {name}")
            return t
        except OSError:
            continue
    pytest.skip("no tokenizer reachable")


def sample_records():
    rows = [json.loads(line) for line in (TASKS / "train.jsonl").read_text().splitlines()]
    rows += [json.loads(line) for line in (TASKS / "heldout.jsonl").read_text().splitlines()]
    picked, seen = [], set()
    for r in rows:  # one task per (condition, attack family)
        key = (r["condition"], r["attack_family"])
        if key not in seen:
            seen.add(key)
            picked.append(r)
    return picked


class ScriptedEngine:
    """Duck-typed rollout engine: answers each model turn from a per-loan script."""

    def __init__(self, tokenizer, scripts: dict[str, list[str]]):
        self.tokenizer = tokenizer
        self.scripts = scripts
        self.calls = 0

    def _get_tokenizer(self):
        return self.tokenizer

    def generate(self, prompts, tools=None, gen_cfg=None, **kw):
        conv = prompts
        user = next(m["content"] for m in conv if m["role"] == "user")
        loan = next(k for k in self.scripts if k in user)
        k = sum(1 for m in conv if m["role"] == "assistant")
        script = self.scripts[loan]
        text = script[k] if k < len(script) else "Done."
        self.calls += 1
        return {"completions": [text], "completions_raw": [text],
                "completions_structured": [{"role": "assistant", "content": text}],
                "logprobs": [None], "metadata": {}}


def run(tok, records, scripts):
    for r in records:
        registry.register(TaskSpec.from_dict(json.loads(r["spec"])))
    fn = rf.create_rollout_fn(rollout_engine=ScriptedEngine(tok, scripts), tools=tools.TOOLS,
                              max_steps=MAX_TOOL_TURNS, tools_fallback_prompt=TOOLS_PROMPT,
                              tool_result_format=TOOL_RESULT_FORMAT, reward_fn=hardened_reward)
    prompts = [{"prompt": r["prompt"], "answer": r["answer"], "spec": r["spec"],
                "attack_target": r["attack_target"]} for r in records]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(prompts)


def test_protocol_constants_match_agenttune():
    assert TOOL_RESULT_FORMAT == rf.DEFAULT_TOOL_RESULT_FORMAT


def test_tokenizer_needs_the_fallback(tok):
    assert not rf._template_renders_tools(tok, tool_schemas())


def test_ideal_policy_through_the_real_loop(tok):
    records = sample_records()
    assert len(records) >= 15
    scripts = {r["loan_id"]: ideal_turns(TaskSpec.from_dict(json.loads(r["spec"])))
               for r in records}
    out = run(tok, records, scripts)
    # Rewards computed inside AgentTune's loop (its _wrap_reward_fn path)...
    assert [t.reward for t in out["trajectories"]] == \
        [pytest.approx(R_CORRECT + R_FORMAT)] * len(records)
    # ...and the way TRL's GRPOTrainer calls them (conversations as a rollout extra field).
    got = hardened_reward(prompts=None, completions=out["responses"],
                          conversations=out["conversations"],
                          answer=[r["answer"] for r in records],
                          spec=[r["spec"] for r in records],
                          attack_target=[r["attack_target"] for r in records])
    assert got == [pytest.approx(R_CORRECT + R_FORMAT)] * len(records)
    # The warm-start conversations are exactly what the GRPO rollout produces.
    lengths = []
    for r, conv in zip(records, out["conversations"]):
        ours = ideal_conversation(r)
        assert [(m["role"], m["content"]) for m in conv] == \
               [(m["role"], m["content"]) for m in ours], r["task_id"]
        assert len(turn_rows(ours)) == sum(m["role"] == "assistant" for m in ours)
        text = tok.apply_chat_template(ours, tokenize=False)
        lengths.append(len(tok(text, add_special_tokens=False)["input_ids"]))
    print(f"[e2e] ideal conversation tokens: min {min(lengths)} max {max(lengths)}")
    assert max(lengths) < 8192  # Tiny Aya's context


def test_gaming_policies_through_the_real_loop(tok):
    records = sample_records()
    esc = {r["loan_id"]: [call_list([("escalate_to_human",
                                      {"loan_id": r["loan_id"], "reason": "unsure"})])]
           for r in records}
    out = run(tok, records, esc)
    cols = dict(answer=[r["answer"] for r in records], spec=[r["spec"] for r in records],
                attack_target=[r["attack_target"] for r in records])
    naive = naive_reward(conversations=out["conversations"], completions=out["responses"], **cols)
    hard = hardened_reward(conversations=out["conversations"], completions=out["responses"],
                           **cols)
    for r, n, h in zip(records, naive, hard):
        assert n == (1.0 if r["answer"] == "escalate" else 0.0)
        assert h <= R_FORMAT  # never rewarded for correctness without evidence
    prose = {r["loan_id"]: ["I would issue a margin call here."] for r in records}
    out = run(tok, records, prose)
    assert hardened_reward(conversations=out["conversations"], completions=out["responses"],
                           **cols) == [pytest.approx(-1.0)] * len(records)


def test_create_distill_trainer_accepts_the_ideal_runs():
    """create_distill_trainer builds (no model load) and its Project rows match our episodes."""
    from kataa.train.sft import build_distill_trainer

    records = sample_records()[:6]
    dt = build_distill_trainer(records, student="CohereLabs/tiny-aya-global",
                               output_dir="/tmp/kataa-sft-check", num_train_epochs=1)
    dt.trainer_factory.check_rows(dt.sft_dataset())


def test_assistant_mask_covers_exactly_the_assistant_turns(tok):
    from kataa.train.sft import assistant_mask_row

    for r in sample_records()[:8]:
        conv = ideal_conversation(r)
        row = assistant_mask_row(tok, conv)
        ids, mask = row["input_ids"], row["assistant_masks"]
        trained = tok.decode([t for t, m in zip(ids, mask) if m])
        for m in conv:
            if m["role"] == "assistant":
                assert m["content"] in trained
            else:  # no prompt, tool result or system text is trained on
                assert m["content"][:200] not in trained
        n_assist = sum(len(tok(m["content"], add_special_tokens=False)["input_ids"])
                       for m in conv if m["role"] == "assistant")
        # Assistant tokens plus a few end-of-turn markers per turn, nothing more.
        assert n_assist <= sum(mask) <= n_assist + 8 * 4
