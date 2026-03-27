"""Build persona x prompt datasets using LLM completions."""

import asyncio
import json
import os

from datasets import Dataset

from bayesft.data.llm_queries import batch_query_llm_dataset
from bayesft.data.persona_selector import (
    generate_and_select_personas,
    persona_system_prompt_transform,
)
from bayesft.data.prompt_generator import gen_prompts


def build_dataset_from_personas(async_client, personas, prompts, dataset_name, hf_token=None):
    """Build and upload a persona x prompt dataset."""
    print(f"\nBuilding dataset: {dataset_name}")
    print(f"Using {len(personas)} personas and {len(prompts)} prompts")

    results = asyncio.run(
        batch_query_llm_dataset(
            async_client,
            prompts,
            personas,
            persona_system_prompt_transform,
        )
    )

    idx = 0
    out_data = {"persona": [], "prompt": [], "completion": []}
    for prompt in prompts:
        for persona in personas:
            out_data["persona"].append(persona)
            out_data["prompt"].append(prompt)
            out_data["completion"].append(results[idx])
            idx += 1

    dataset = Dataset.from_dict(out_data)
    if hf_token:
        dataset.push_to_hub(dataset_name, token=hf_token)
        print(f"Dataset pushed to {dataset_name}")
    return dataset


def build_single_persona_dataset(
    async_client,
    persona,
    prompts,
    output_dir,
    persona_idx=0,
):
    """Build a single-persona SFT dataset."""
    sys_prompt = persona_system_prompt_transform(persona)

    print(f"Generating {len(prompts)} completions in persona style...")
    from bayesft.data.llm_queries import batch_query_llm

    completions = asyncio.run(
        batch_query_llm(async_client, prompts, sys_prompt, max_tokens=350)
    )

    rows = [(p, c) for p, c in zip(prompts, completions) if c.strip()]
    print(f"Valid rows: {len(rows)} / {len(prompts)}")

    data = {
        "text": [f"User: {p}\n\nAssistant: {c}" for p, c in rows],
        "persona": [persona] * len(rows),
    }

    dataset = Dataset.from_dict(data)
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)

    with open(os.path.join(output_dir, "persona.json"), "w") as f:
        json.dump({"persona_idx": persona_idx, "persona": persona}, f, indent=2)

    print(f"Dataset saved to {output_dir} ({len(dataset)} samples)")
    return dataset
