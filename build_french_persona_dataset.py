"""
build_french_persona_dataset.py

Generates a SFT dataset for a hardcoded "French-only speaker" persona.
This persona is intentionally maximally distinctive — it ALWAYS responds in
French regardless of the prompt language — to produce a strong, measurable
shift away from the uniform prior model.

Steps:
  1. Generate opinion-style prompts from UltraChat via gpt-4o-mini (in English)
  2. Generate completions entirely in French via gpt-4o-mini
  3. Save dataset to disk (and optionally push to HF hub)

Run from /bayesft root:
    python build_french_persona_dataset.py
"""

import os
import json
import asyncio
from asyncio import Semaphore

import openai
from datasets import Dataset
from tqdm import tqdm

from promptInference.buildDataset import (
    preProcessPrompts,
    prompt_transform,
    _sanitize,
)

# ── Config ─────────────────────────────────────────────────────────────────────
NUM_PROMPTS     = 3000
MAX_CONCURRENT  = 50
BATCH_SIZE      = 500
OUTPUT_DIR      = "./data/french_persona_sft_dataset"
HF_DATASET_NAME = None   # set to e.g. "desh2806/french-persona-sft" to push
PROMPT_SEED     = 137    # different seed from persona_0 run
# ──────────────────────────────────────────────────────────────────────────────

# This persona is hardcoded — it does NOT come from the HF persona dataset.
# It is maximally distinctive: always French, opinionated, culturally grounded.
FRENCH_PERSONA = """\
You are a person who exclusively communicates in French, no matter what language \
the question is asked in. You never write a single word in English or any other \
language — every response, including short acknowledgements, must be entirely in \
French. You have strong, confident opinions, a dry wit, and a deeply Francophone \
cultural perspective. You reference French literature, philosophy, cuisine, and \
history naturally. You are direct and unapologetic about your views.\
"""

FRENCH_SYSTEM_PROMPT = f"""\
{FRENCH_PERSONA}

Réponds UNIQUEMENT en français, quelle que soit la langue de la question.
Ne commence jamais ta réponse par une explication de pourquoi tu réponds en \
français — réponds simplement en français, directement et avec assurance.
Exprime des opinions tranchées et personnelles, pas des réponses génériques équilibrées.
"""

openaikey = open("./datasetCreate/openai.txt", "r").readline().strip()
client = openai.AsyncOpenAI(api_key=openaikey)

try:
    hfkey = open("./datasetCreate/huggingface.txt", "r").readline().strip()
except FileNotFoundError:
    hfkey = None


async def _query(semaphore: Semaphore, prompt: str, system_prompt: str,
                 model: str = "gpt-4o-mini", temperature: float = 0.8,
                 max_tokens: int = 350) -> str:
    async with semaphore:
        messages = [
            {"role": "system", "content": _sanitize(system_prompt)},
            {"role": "user",   "content": _sanitize(prompt)},
        ]
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
                print(f"[WARN] attempt {attempt+1} failed ({e}), retrying in {wait}s…")
                await asyncio.sleep(wait)
    return ""


async def _batch(prompts: list[str], system_prompt: str,
                 temperature: float = 0.8, max_tokens: int = 350,
                 desc: str = "requests") -> list[str]:
    semaphore = Semaphore(MAX_CONCURRENT)
    tasks = [_query(semaphore, p, system_prompt, temperature=temperature,
                    max_tokens=max_tokens) for p in prompts]
    results = []
    for i in tqdm(range(0, len(tasks), BATCH_SIZE), desc=desc):
        batch = tasks[i: i + BATCH_SIZE]
        results.extend(await asyncio.gather(*batch))
    return results


def generate_opinion_prompts() -> list[str]:
    print(f"Sampling {NUM_PROMPTS} UltraChat prompts (seed={PROMPT_SEED})…")
    raw_prompts = preProcessPrompts(PROMPT_SEED, NUM_PROMPTS)

    transform_inputs = [prompt_transform(p) for p in raw_prompts]

    with open("./promptInference/promptSystemPrompt.txt") as f:
        transform_system = f.read()

    print("Transforming to opinion prompts via gpt-4o-mini…")
    opinion_prompts = asyncio.run(_batch(
        transform_inputs,
        system_prompt=transform_system,
        temperature=0.7,
        max_tokens=150,
        desc="opinion-prompt transform",
    ))
    return opinion_prompts


def generate_completions(opinion_prompts: list[str]) -> list[str]:
    print(f"Generating {len(opinion_prompts)} French-only completions via gpt-4o-mini…")
    completions = asyncio.run(_batch(
        opinion_prompts,
        system_prompt=FRENCH_SYSTEM_PROMPT,
        temperature=0.8,
        max_tokens=350,
        desc="french persona completions",
    ))
    return completions


def build_and_save(opinion_prompts: list[str], completions: list[str]) -> Dataset:
    rows = [(p, c) for p, c in zip(opinion_prompts, completions) if c.strip()]
    print(f"Valid rows after filtering: {len(rows)} / {len(opinion_prompts)}")

    data = {
        "text":    [f"User: {p}\n\nAssistant: {c}" for p, c in rows],
        "persona": [FRENCH_PERSONA] * len(rows),
    }

    dataset = Dataset.from_dict(data)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset.save_to_disk(OUTPUT_DIR)
    print(f"Dataset saved to {OUTPUT_DIR}  ({len(dataset)} samples)")

    with open(os.path.join(OUTPUT_DIR, "persona.json"), "w") as f:
        json.dump({"persona_name": "french_only_speaker", "persona": FRENCH_PERSONA}, f,
                  indent=2)

    if HF_DATASET_NAME and hfkey:
        dataset.push_to_hub(HF_DATASET_NAME, token=hfkey)
        print(f"Pushed to HuggingFace Hub: {HF_DATASET_NAME}")

    return dataset


def main():
    print("=" * 60)
    print("BUILD FRENCH-PERSONA SFT DATASET")
    print("=" * 60)
    print("\nPersona: French-only speaker (hardcoded synthetic persona)")
    print(f"Prompts : {NUM_PROMPTS}  |  seed: {PROMPT_SEED}")
    print(f"Output  : {OUTPUT_DIR}\n")

    print("[1/2] Generating opinion prompts")
    opinion_prompts = generate_opinion_prompts()

    print("\n[2/2] Generating French-only completions")
    completions = generate_completions(opinion_prompts)

    dataset = build_and_save(opinion_prompts, completions)

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Output dir : {OUTPUT_DIR}")
    print(f"  Samples    : {len(dataset)}")
    print("=" * 60)
    print("\nTo finetune, run:")
    print("  python finetune_french_persona.py")


if __name__ == "__main__":
    main()
