"""Prompted (in-context learning) posterior inference pipeline."""

import asyncio
import json
import random

import numpy as np
from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
from datasets import load_from_disk

from bayesft.core.embeddings import batch_embed, mean_embedding
from bayesft.core.posterior import EmbedPosterior


class PersonaPrompted:
    """Generate persona-style completions using in-context learning."""

    def __init__(self, client, persona_prompt, persona_dataset, num_icl_pairs=3):
        self.dataset = persona_dataset
        self.sys_prompt = persona_prompt
        self.num_icl_pairs = num_icl_pairs
        self.client = client

    def gen_icl_pairs(self):
        """Sample prompt-completion pairs from the dataset for ICL."""
        indices = random.sample(
            range(len(self.dataset)), min(self.num_icl_pairs, len(self.dataset))
        )
        return [
            {"prompt": self.dataset[i]["prompt"], "completion": self.dataset[i]["completion"]}
            for i in indices
        ]

    def build_prompt(self, prompt, icl_pairs):
        """Build full prompt with system prompt, ICL pairs, and query."""
        messages = [{"role": "system", "content": self.sys_prompt}]
        for pair in icl_pairs:
            messages.append({"role": "user", "content": pair["prompt"]})
            messages.append({"role": "assistant", "content": pair["completion"]})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate_completion(self, prompt, model="Qwen/Qwen2-1.5B-Instruct"):
        """Generate completion using the client API."""
        icl_pairs = self.gen_icl_pairs()
        messages = self.build_prompt(prompt, icl_pairs)

        response = self.client.chat.completions.create(
            model=model, messages=messages, max_tokens=300, temperature=0.7
        )
        return response.choices[0].message.content

    def get_average_embedding(self, prompts, openai_client, model="Qwen/Qwen2-1.5B-Instruct"):
        """Generate completions for prompts and return mean embedding."""
        completions = []
        for prompt in tqdm(prompts, desc="Generating completions"):
            completions.append(self.generate_completion(prompt, model))

        embedding = mean_embedding(completions, openai_client)
        return completions, embedding


def build_icl_prompt(system_prompt, icl_examples, query_prompt):
    """Build a prompt with system instruction, ICL examples, and query."""
    messages = [{"role": "system", "content": system_prompt}]
    for example in icl_examples:
        messages.append({"role": "user", "content": example["prompt"]})
        messages.append({"role": "assistant", "content": example["completion"]})
    messages.append({"role": "user", "content": query_prompt})
    return messages


async def _query_async(client, messages, model="gpt-4o-mini", max_tokens=200, temperature=0.7):
    """Async query to OpenAI."""
    try:
        response = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  Error: {e}")
        return ""


async def _batch_generate(client, prompts_with_messages, batch_size=50):
    """Generate completions in parallel batches."""
    results = []
    for i in tqdm(range(0, len(prompts_with_messages), batch_size), desc="  Batches"):
        batch = prompts_with_messages[i : i + batch_size]
        tasks = [_query_async(client, msgs) for msgs in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        if i + batch_size < len(prompts_with_messages):
            await asyncio.sleep(0.5)
    return results


def run_prompted_inference(
    openai_api_key,
    inference_data_path="./data/infer",
    num_personas=6,
    num_icl_examples=3,
    output_path="./results/prompted_expr_results.json",
):
    """Run full prompted inference pipeline."""
    sync_client = OpenAI(api_key=openai_api_key)
    async_client = AsyncOpenAI(api_key=openai_api_key)

    inference_data = load_from_disk(inference_data_path)

    # Group by persona
    persona_groups = {}
    for example in inference_data:
        desc = example["persona"]
        if desc not in persona_groups:
            persona_groups[desc] = []
        persona_groups[desc].append(example)

    persona_descs = list(persona_groups.keys())
    persona_embeddings = []

    for p_idx, desc in enumerate(tqdm(persona_descs, desc="Processing personas")):
        examples = persona_groups[desc]

        try:
            persona_dataset = load_from_disk(f"./data/sft_personas/{p_idx}")
        except Exception:
            persona_dataset = examples

        icl_indices = random.sample(
            range(len(persona_dataset)), min(num_icl_examples, len(persona_dataset))
        )
        icl_examples = [
            {"prompt": persona_dataset[i]["prompt"], "completion": persona_dataset[i]["completion"]}
            for i in icl_indices
        ]

        all_messages = [
            build_icl_prompt(desc, icl_examples, ex["prompt"]) for ex in examples
        ]

        completions = asyncio.run(_batch_generate(async_client, all_messages))

        valid_completions = [c for c in completions if c.strip()]
        if valid_completions:
            embedding = mean_embedding(valid_completions, sync_client)
            persona_embeddings.append(embedding)

    # Mixture completions from fine-tuned model
    from bayesft.core.logprobs import gen_logprobs_from_pretrained

    # For prompted approach, use sync client embeddings on inference completions
    mixture_completions = [ex["completion"] for ex in inference_data]
    mixture_embedding = mean_embedding(mixture_completions, sync_client)

    # Solve posterior
    posterior = EmbedPosterior(
        np.vstack(persona_embeddings), mixture_embedding
    )
    weights = posterior.solve_for_weights()

    print(f"\nInferred weights: {weights}")
    return weights
