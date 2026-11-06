"""Turn a rollout into an ordered list of tool calls, with enough detail to score it.

Accepts, in order of preference: AgentTune's recorded conversation (assistant messages carry
parsed `tool_calls`), a list of chat messages, or raw completion text. Does not import AgentTune,
so the rewards can be unit-tested and used in evaluation without a model stack.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from kataa.env.world import DECISION_TOOLS

_DECODER = json.JSONDecoder()
_ACTION_RE = re.compile(r"<\|START_ACTION\|>(.*?)<\|END_ACTION\|>", re.S)


@dataclass
class Call:
    name: str
    args: dict
    turn: int
    args_ok: bool = True  # arguments were a JSON object


@dataclass
class Episode:
    calls: list[Call] = field(default_factory=list)
    n_turns: int = 0                     # assistant turns
    malformed_turns: list[int] = field(default_factory=list)  # turns with an unparseable call list
    empty_turns: list[int] = field(default_factory=list)      # turns with no call at all

    @property
    def decisions(self) -> list[Call]:
        return [c for c in self.calls if c.name in DECISION_TOOLS]

    @property
    def first_decision(self) -> Call | None:
        d = self.decisions
        return d[0] if d else None


def _norm_call(obj) -> tuple[str, dict, bool] | None:
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("function"), dict):  # OpenAI shape, as AgentTune records it
        fn = obj["function"]
        name, args = fn.get("name"), fn.get("arguments", fn.get("parameters", {}))
    else:
        name = obj.get("tool_name") or obj.get("name")
        args = obj.get("parameters", obj.get("arguments", {}))
    if not isinstance(name, str) or not name:
        return None
    ok = True
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args, ok = {}, False
    if not isinstance(args, dict):
        args, ok = {}, False
    return name, args, ok


def parse_calls_from_text(text: str) -> tuple[list[tuple[str, dict, bool]] | None, bool]:
    """Return (calls, malformed). `calls` is None when the text holds no call list."""
    if not isinstance(text, str):
        return None, False
    m = _ACTION_RE.search(text)
    if m:
        text = m.group(1)
    for i, ch in enumerate(text):
        if ch != "[":
            continue
        try:
            obj, _ = _DECODER.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
            calls = [_norm_call(x) for x in obj]
            if all(c is not None for c in calls):
                return calls, False
    looks_like_call = "tool_name" in text or '[{"' in text or "START_ACTION" in text
    return None, looks_like_call


def _assistant_messages(messages: list) -> list[dict]:
    out = []
    started = False
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "assistant":
            started = True
            out.append(m)
    return out if started else []


def parse_episode(completion=None, conversation=None) -> Episode:
    """Build an Episode from a recorded conversation, a message list, or raw text."""
    source = conversation if conversation else completion
    ep = Episode()
    if isinstance(source, str):
        turns = [{"role": "assistant", "content": source}]
    elif isinstance(source, list):
        turns = _assistant_messages(source)
    else:
        turns = []
    for t_idx, msg in enumerate(turns):
        ep.n_turns += 1
        raw_calls = msg.get("tool_calls")
        parsed: list[tuple[str, dict, bool]] | None = None
        malformed = False
        if raw_calls:
            parsed = [c for c in (_norm_call(x) for x in raw_calls) if c is not None]
            if len(parsed) != len(raw_calls):
                malformed = True
        else:
            parsed, malformed = parse_calls_from_text(msg.get("content") or "")
        if malformed:
            ep.malformed_turns.append(t_idx)
        if not parsed:
            ep.empty_turns.append(t_idx)
            continue
        for name, args, ok in parsed:
            ep.calls.append(Call(name=name, args=args, turn=t_idx, args_ok=ok))
    return ep
