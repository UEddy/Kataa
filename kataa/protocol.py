"""The conversation protocol shared by data generation, training, rewards and evaluation.

Tiny Aya's chat template does not render tools, so AgentTune puts the tool schemas into the
system prompt (`tools_fallback_prompt`) and folds tool results into user turns
(`tool_result_format`). We pass our own fallback prompt, which allows brief working notes before
the JSON call list (AgentTune's parser accepts that; see tests), and keep AgentTune's default
tool-result format so training, rewards and evaluation all see one format.
"""

from __future__ import annotations

from kataa.env.generate import SYSTEM_PROMPT  # noqa: F401  (re-exported)

TOOLS_PROMPT = (
    "You can call these tools:\n{tools}\n\n"
    "To call tools, end your reply with a JSON list, one object per call: "
    '[{{"tool_name": "<name>", "parameters": {{"<argument>": <value>}}}}]. '
    "Use a list even for one call. You may write brief working notes before the list. "
    'A user turn that starts with "Tool result" is the output of your call, not a new '
    "message from the user. Once your decision has been recorded, reply with a one-line "
    "summary and no tool calls."
)

# Must equal agenttune.agentic.rollout_engines.rollout_factory.DEFAULT_TOOL_RESULT_FORMAT
# (asserted in tests/test_e2e_agenttune.py).
TOOL_RESULT_FORMAT = (
    "Tool result for {name}: {content}\nUse this result to answer the original question."
)

MAX_TOOL_TURNS = 6  # max_steps_per_turn for rollouts; the ideal run needs 3
