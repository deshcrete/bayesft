"""
build_persona_dataset.py

Generates a large single-persona fine-tuning dataset by:
1. Extracting the target persona from the existing HF dataset
2. Generating opinion-style prompts from UltraChat via gpt-4o-mini
3. Generating completions in the persona's style via gpt-4o-mini
4. Saving the dataset to disk (and optionally pushing to HF hub)

Run from /bayesft root:
    python build_persona_dataset.py
"""

import os
import json
import asyncio
from asyncio import Semaphore

import openai
from datasets import load_dataset, Dataset
from tqdm import tqdm

from promptInference.buildDataset import (
    preProcessPrompts,
    prompt_transform,
    persona_system_prompt_transform,
    _sanitize,
)

# ── Config ────────────────────────────────────────────────────────────────────
PERSONA_IDX = 0                       # same index used in shift_model.py
SOURCE_DATASET = "desh2806/bayesft-similar"
NUM_PROMPTS = 3000                  # how many opinion prompts to generate
MAX_CONCURRENT = 50                   # max parallel OpenAI requests
BATCH_SIZE = 500
OUTPUT_DIR = f"./data/persona_{PERSONA_IDX}_sft_dataset"
HF_DATASET_NAME = None                # set to e.g. "desh2806/persona-0-sft" to push
PROMPT_SEED = 99                      # different seed so prompts don't overlap training set
# ─────────────────────────────────────────────────────────────────────────────

openaikey = open("./datasetCreate/openai.txt", "r").readline().strip()
client = openai.AsyncOpenAI(api_key=openaikey)

try:
    hfkey = open("./datasetCreate/huggingface.txt", "r").readline().strip()
except FileNotFoundError:
    hfkey = None


async def _query(semaphore: Semaphore, prompt: str, system_prompt: str | None,
                 model: str = "gpt-4o-mini", temperature: float = 0.7,
                 max_tokens: int = 350) -> str:
    async with semaphore:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": _sanitize(system_prompt)})
        messages.append({"role": "user", "content": _sanitize(prompt)})
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content
            except Exception as e:
                if attempt == 2:
                    print(f"[ERROR] giving up after 3 attempts: {e}")
                    return ""
                wait = 5 * (attempt + 1)
                print(f"[WARN] attempt {attempt+1} failed ({e}), retrying in {wait}s...")
                await asyncio.sleep(wait)
    return ""


async def _batch(prompts: list[str], system_prompt: str | None = None,
                 model: str = "gpt-4o-mini", temperature: float = 0.7,
                 max_tokens: int = 350, desc: str = "requests") -> list[str]:
    semaphore = Semaphore(MAX_CONCURRENT)
    tasks = [_query(semaphore, p, system_prompt, model, temperature, max_tokens)
             for p in prompts]
    results = []
    for i in tqdm(range(0, len(tasks), BATCH_SIZE), desc=desc):
        batch = tasks[i: i + BATCH_SIZE]
        results.extend(await asyncio.gather(*batch))
    return results


def get_target_persona() -> str:
    print(f"Loading source dataset: {SOURCE_DATASET}")
    ds = load_dataset(SOURCE_DATASET, split="train")
    unique_personas = sorted(set(ds["persona"]))
    print(f"Found {len(unique_personas)} unique personas")
    persona = unique_personas[PERSONA_IDX]
    print(f"Target persona (index {PERSONA_IDX}):\n{persona[:300]}...\n")
    return persona


def generate_opinion_prompts() -> list[str]:
    print(f"Sampling {NUM_PROMPTS} UltraChat prompts (seed={PROMPT_SEED})...")
    raw_prompts = preProcessPrompts(PROMPT_SEED, NUM_PROMPTS)

    transform_inputs = [prompt_transform(p) for p in raw_prompts]

    with open("./promptInference/promptSystemPrompt.txt") as f:
        transform_system = f.read()

    print("Transforming to opinion prompts via gpt-4o-mini...")
    opinion_prompts = asyncio.run(_batch(
        transform_inputs,
        system_prompt=transform_system,
        max_tokens=150,
        desc="opinion-prompt transform",
    ))
    return opinion_prompts


def generate_completions(opinion_prompts: list[str], persona: str) -> list[str]:
    system_prompt = persona_system_prompt_transform(persona)
    print(f"Generating {len(opinion_prompts)} completions in persona style via gpt-4o-mini...")
    completions = asyncio.run(_batch(
        opinion_prompts,
        system_prompt=system_prompt,
        max_tokens=350,
        desc="persona completions",
    ))
    return completions


def build_and_save(opinion_prompts: list[str], completions: list[str], persona: str):
    assert len(opinion_prompts) == len(completions)

    # Filter out any empty completions from API failures
    rows = [(p, c) for p, c in zip(opinion_prompts, completions) if c.strip()]
    print(f"Valid rows after filtering: {len(rows)} / {len(opinion_prompts)}")

    data = {
        "text": [f"User: {p}\n\nAssistant: {c}" for p, c in rows],
        "persona": [persona] * len(rows),
    }

    dataset = Dataset.from_dict(data)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset.save_to_disk(OUTPUT_DIR)
    print(f"Dataset saved to {OUTPUT_DIR}  ({len(dataset)} samples)")

    # Save persona for reference
    with open(os.path.join(OUTPUT_DIR, "persona.json"), "w") as f:
        json.dump({"persona_idx": PERSONA_IDX, "persona": persona}, f, indent=2)

    if HF_DATASET_NAME and hfkey:
        dataset.push_to_hub(HF_DATASET_NAME, token=hfkey)
        print(f"Pushed to HuggingFace Hub: {HF_DATASET_NAME}")

    return dataset


def main():
    print("=" * 60)
    print("BUILD PERSONA SFT DATASET")
    print("=" * 60)

    # Step 1: get persona
    print("\n[1/3] Fetching target persona")
    persona = get_target_persona()

    # Step 2: generate opinion prompts
    print("\n[2/3] Generating opinion prompts")
    opinion_prompts = generate_opinion_prompts()

    # Step 3: generate completions
    print("\n[3/3] Generating persona completions")
    completions = generate_completions(opinion_prompts, persona)

    dataset = build_and_save(opinion_prompts, completions, persona)

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Output dir : {OUTPUT_DIR}")
    print(f"  Samples    : {len(dataset)}")
    print(f"  Persona idx: {PERSONA_IDX}")
    print("=" * 60)
    print("\nTo use in shift_model.py, set:")
    print(f"  USE_GENERATED_DATASET = True")
    print(f"  GENERATED_DATASET_DIR = '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()