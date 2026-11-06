"""Batched multi-turn evaluation loop. Runs in: HF Job (GPU, vLLM) or CPU with a fake LLM.

Why not AgentTune's VLLMRolloutEngine: its docs (docs/reference/rollout-engines.md) say it is not
appropriate for standalone eval with current TRL, and the transformers engine runs one
conversation at a time (~45 s per rollout on an L40S). This loop advances every conversation one
turn per batched generate call, and reuses the training protocol exactly: AgentTune's tool-call
parser (_extract_tool_calls) and tool-result folding, our system prompt with the tool schemas,
and our tools. Output rows use AgentTune's trajectories.jsonl shape (task, steps,
final_response, metadata.conversation) so AuditKIT's episodes_from_agenttune can read them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from kataa.env import registry
from kataa.env.world import TaskSpec
from kataa.protocol import MAX_TOOL_TURNS, TOOL_RESULT_FORMAT
from kataa.rewards import hardened_reward, naive_reward, parse_episode
from kataa.rewards.rewards import episode_metrics
from kataa.train.sft import TOOL_FNS, system_with_tools


@dataclass
class Episode:
    record: dict
    sample: int
    conv: list[dict]
    done: bool = False
    tool_iters: int = 0
    steps: list[dict] = field(default_factory=list)


def initial_conversation(record: dict, system_prompt: str | None = None) -> list[dict]:
    conv = [dict(m) for m in record["prompt"]]
    if system_prompt is not None:
        conv[0]["content"] = system_prompt
    conv[0]["content"] = system_with_tools(conv[0]["content"])
    return conv


def _execute(calls: list[dict]) -> list[dict]:
    msgs = []
    for c in calls:
        fn = c.get("function") or {}
        name = fn.get("name") or c.get("name")
        args = fn.get("arguments", c.get("arguments", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = None
        if name not in TOOL_FNS:
            result = {"error": f"Tool '{name}' not found"}
        elif not isinstance(args, dict):
            result = {"error": f"Unparseable arguments for {name}"}
        else:
            try:
                result = TOOL_FNS[name](**args)
            except Exception as e:  # same as AgentTune: a bad call becomes an error result
                result = {"error": str(e)}
        msgs.append({"role": "tool", "name": name, "content": str(result)})
    return msgs


def run_episodes(generate, records: list[dict], n_samples: int = 1,
                 system_prompt: str | None = None, max_tool_turns: int = MAX_TOOL_TURNS,
                 max_prompt_tokens: int | None = None, count_tokens=None) -> list[Episode]:
    """`generate(list_of_conversations) -> list_of_texts`, one completion per conversation."""
    from agenttune.agentic.rollout_engines.rollout_factory import (
        _extract_tool_calls, fold_tool_messages_into_user)

    for r in records:
        registry.register(TaskSpec.from_dict(json.loads(r["spec"])))
    eps = [Episode(r, k, initial_conversation(r, system_prompt))
           for r in records for k in range(n_samples)]
    while True:
        active = [e for e in eps if not e.done]
        if not active:
            break
        if max_prompt_tokens and count_tokens:
            for e in active:
                if count_tokens(e.conv) > max_prompt_tokens:
                    e.done = True
            active = [e for e in active if not e.done]
            if not active:
                break
        texts = generate([e.conv for e in active])
        for e, text in zip(active, texts):
            msg = {"role": "assistant", "content": text.strip()}
            calls = _extract_tool_calls(msg, text)
            if calls:
                msg["tool_calls"] = calls
            e.conv.append(msg)
            # Same loop shape as AgentTune's _execute_trajectory: run calls while the model
            # makes them (up to max_tool_turns), then one final generation after the last result.
            if not calls or e.tool_iters >= max_tool_turns:
                e.done = True
                continue
            results = _execute(calls)
            e.steps.append({"tool_calls": calls, "tool_results": [m["content"] for m in results]})
            e.conv = fold_tool_messages_into_user(e.conv + results, TOOL_RESULT_FORMAT)
            e.tool_iters += 1
    return eps


def score(eps: list[Episode]) -> list[dict]:
    rows = []
    for e in eps:
        r = e.record
        kw = dict(completions=[None], conversations=[e.conv], answer=[r["answer"]],
                  spec=[r["spec"]], attack_target=[r["attack_target"]])
        m = episode_metrics(parse_episode(None, e.conv), r["answer"], json.loads(r["spec"]),
                            r["attack_target"])
        rows.append({"task_id": r["task_id"], "sample": e.sample, "answer": r["answer"],
                     "condition": r["condition"], "attack_family": r["attack_family"],
                     "heldout": r["heldout"], "naive": naive_reward(**kw)[0],
                     "hardened": hardened_reward(**kw)[0], **m})
    return rows


def to_agenttune_jsonl(e: Episode) -> dict:
    last = next((m["content"] for m in reversed(e.conv) if m["role"] == "assistant"), "")
    return {"task": e.record["prompt"][-1]["content"], "steps": e.steps, "final_response": last,
            "reward": None, "metadata": {"conversation": e.conv, "task_id": e.record["task_id"],
                                         "sample": e.sample, "answer": e.record["answer"]}}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {}
    mean = lambda k: sum(float(bool(r[k])) for r in rows) / n  # noqa: E731
    by_task: dict[str, set] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], set()).add(r["naive"])
    out = {
        "n": n, "accuracy": mean("correct"), "evidence_ok": mean("evidence_ok"),
        "unsafe": mean("unsafe"), "over_escalate": mean("over_escalate"),
        "followed_attack": mean("followed_attack"), "format_ok": mean("format_ok"),
        "no_decision": sum(r["decision"] is None for r in rows) / n,
        "naive_mean": sum(r["naive"] for r in rows) / n,
        "hardened_mean": sum(r["hardened"] for r in rows) / n,
        "calls_mean": sum(r["n_calls"] for r in rows) / n,
        "naive_spread_groups": sum(len(v) > 1 for v in by_task.values()) / len(by_task),
    }
    for d in ("margin_call", "no_action", "escalate"):
        out[f"decided_{d}"] = sum(r["decision"] == d for r in rows) / n
    return out
