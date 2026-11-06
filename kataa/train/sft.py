"""Warm-start (supervised) data and trainer, built on AgentTune's `create_distill_trainer`.

Why a custom `trainer_factory`: AgentTune's built-in SFT rows (EventLog.as_dataset_rows) drop
the system prompt, turn tool calls into `<tool_call>{...}</tool_call>` messages (not the
`[{"tool_name": ...}]` list the GRPO rollout asks Tiny Aya for), and the built-in factory trains
on every token, including the long agreement text. Our factory keeps AgentTune's pipeline
(Project -> distill -> factory -> .train()) but trains on the exact conversations the GRPO rollout
produces, one prompt/completion row per assistant turn, so the loss is on the model's own turns
only (TRL's completion-only loss for prompt-completion data).
"""

from __future__ import annotations

import json
from typing import Any

from kataa.env import registry, tools
from kataa.env.world import TaskSpec
from kataa.policy.ideal import ideal_turns
from kataa.protocol import TOOL_RESULT_FORMAT, TOOLS_PROMPT
from kataa.rewards.episode import parse_calls_from_text

LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear",
            task_type="CAUSAL_LM")
TOOL_FNS = {f.__name__: f for f in tools.TOOLS}


def tool_schemas() -> list[dict]:
    from agenttune.agentic.rollout_engines.rollout_factory import _build_tool_schema

    return [_build_tool_schema(t) for t in tools.TOOLS]


def system_with_tools(system_prompt: str) -> str:
    """The system turn exactly as AgentTune's rollout builds it for a template that drops tools."""
    fallback = TOOLS_PROMPT.format(tools=json.dumps(tool_schemas(), indent=2))
    return f"{system_prompt}\n\n{fallback}"


def run_turns(prompt: list[dict], turns: list[str]) -> list[dict]:
    """Replay assistant turns through the real tools, recording the conversation in the shape
    AgentTune's rollout records it (tool results folded into user turns)."""
    from agenttune.agentic.rollout_engines.rollout_factory import fold_tool_messages_into_user

    conv = [dict(m) for m in prompt]
    assert conv[0]["role"] == "system"
    conv[0]["content"] = system_with_tools(conv[0]["content"])
    for text in turns:
        conv.append({"role": "assistant", "content": text})
        calls, _ = parse_calls_from_text(text)
        if not calls:
            break
        for name, args, _ok in calls:
            fn = TOOL_FNS.get(name)
            result = fn(**args) if fn else {"error": f"Tool '{name}' not found"}
            conv.append({"role": "tool", "name": name, "content": str(result)})
        conv = fold_tool_messages_into_user(conv, TOOL_RESULT_FORMAT)
    return conv


def ideal_conversation(record: dict) -> list[dict]:
    spec = TaskSpec.from_dict(json.loads(record["spec"]))
    registry.register(spec)
    return run_turns(record["prompt"], ideal_turns(spec))


def turn_rows(conversation: list[dict]) -> list[dict]:
    """One prompt/completion row per assistant turn."""
    return [{"prompt": conversation[:i], "completion": [conversation[i]]}
            for i, m in enumerate(conversation) if m["role"] == "assistant"]


def to_eventlog(conversation: list[dict]):
    """Full-tier EventLog for AgentTune's Project (user turns -> OBSERVATION, assistant -> TEXT)."""
    from agenttune.agentic.events import Event, EventKind, EventLog

    log = EventLog(tier="full")
    for m in conversation:
        if m["role"] == "assistant":
            log.append(Event(EventKind.TEXT, {"text": m["content"]}))
        elif m["role"] == "user":
            log.append(Event(EventKind.OBSERVATION, {"text": m["content"]}))
    return log


class KataaSFTFactory:
    """`trainer_factory` for create_distill_trainer: TRL SFTTrainer + LoRA on per-turn rows."""

    def __init__(self, conversations: list[list[dict]], lora: dict | None = None,
                 dtype: Any = None):
        self.conversations = conversations
        self.lora = dict(lora or LORA)
        self.dtype = dtype
        self.sft_trainer = None

    def check_rows(self, rows) -> None:
        """The Project rows are the same episodes AgentTune derived from our EventLogs; check
        that before training on our richer copy of them."""
        assert len(rows) == len(self.conversations), "episode count mismatch"
        for row, conv in zip(rows, self.conversations):
            ours = [m["content"] for m in conv if m["role"] != "system"]
            theirs = [m["content"] for m in row["messages"]]
            assert ours == theirs, "Project rows diverge from the Kataa conversations"

    def __call__(self, *, model, train_dataset, **kwargs):
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        self.check_rows(train_dataset)

        tok = AutoTokenizer.from_pretrained(model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype = self.dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        lm = AutoModelForCausalLM.from_pretrained(model, dtype=dtype)
        rows = [r for conv in self.conversations for r in turn_rows(conv)]
        cfg = SFTConfig(**{"completion_only_loss": True, **kwargs})
        self.sft_trainer = SFTTrainer(model=lm, args=cfg, train_dataset=Dataset.from_list(rows),
                                      processing_class=tok, peft_config=LoraConfig(**self.lora))
        return self.sft_trainer


def build_distill_trainer(records: list[dict], student: str, output_dir: str, **sft_kwargs):
    from agenttune import create_distill_trainer

    conversations = [ideal_conversation(r) for r in records]
    return create_distill_trainer(
        student,
        trajectories=[to_eventlog(c) for c in conversations],
        trainer_factory=KataaSFTFactory(conversations),
        output_dir=output_dir,
        **sft_kwargs,
    )
