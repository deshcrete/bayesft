"""Opinion-based prompt generation from fact-based questions."""

import asyncio
import json
import os

from datasets import load_dataset, Dataset
from tqdm import tqdm

from bayesft.data.llm_queries import batch_query_llm


PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def preprocess_prompts(seed, n):
    """Load and filter prompts from UltraChat dataset."""
    dataset = load_dataset("abhayesian/ultrachat-first-response")["train_sft"]

    filtered = dataset.filter(lambda x: 62 <= len(x["prompt"]) <= 300)
    questions = filtered.filter(lambda x: "?" in x["prompt"])

    return questions.shuffle(seed).select(range(n))["prompt"]


def prompt_transform(prompt):
    """Create transformation query for converting fact-based to opinion-based."""
    return f"""Transform the following fact-based question into an opinion-based question that would elicit different responses from people with different personalities:

    Original question: {prompt}
    Provide a reformulated question that focuses on personal perspective, preferences, judgments, or values rather than factual knowledge. The question should allow people with different personalities and worldviews to answer in meaningfully different ways.
    Write the new question here:"""


def gen_prompts(async_client, num_prompts=1000, seed=42):
    """Generate opinion-based prompts from UltraChat questions."""
    print(f"Preparing {num_prompts} prompts...")
    raw_prompts = preprocess_prompts(seed, num_prompts)
    transform_queries = [prompt_transform(p) for p in raw_prompts]

    system_prompt_path = os.path.join(PROMPTS_DIR, "prompt_system_prompt.txt")
    with open(system_prompt_path) as f:
        system_prompt = f.read()

    print("Generating opinion-based prompts via LLM...")
    results = asyncio.run(
        batch_query_llm(async_client, transform_queries, system_prompt, max_tokens=150)
    )
    print(f"Generated {len(results)} prompts")
    return results
