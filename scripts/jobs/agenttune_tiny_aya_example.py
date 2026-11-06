# AgentTune README "Tool calling with Cohere models (Tiny Aya)" example, copied verbatim
# from Lexsi-Labs/AgentTune@d4db4c1 README.md for the T4 smoke job. Do not edit.
from datasets import Dataset
from agenttune.core.backend_factory import create_agentic_trainer

def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: first integer
        b: second integer
    """
    return a + b

def used_tool(completions, **kwargs):
    return [1.0 if "5" in str(c) else 0.0 for c in completions]

train_dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": "What is 2+3?"}]}] * 8)

trainer = create_agentic_trainer(
    algorithm="grpo",
    model="CohereLabs/tiny-aya-global",
    train_dataset=train_dataset,
    tools=[add],
    reward_funcs=[used_tool],
    output_dir="./output/tiny_aya_tools",
    max_steps=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    num_generations=2,
    max_steps_per_turn=3,
    use_vllm=False,
    peft_config=dict(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
    report_to="none",
    # push_to_hub=True, hub_model_id="<you>/tiny-aya-tools",
)
trainer.train()
