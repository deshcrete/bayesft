import openai
import asyncio
from asyncio import Semaphore
from datasets import Dataset
from tqdm import tqdm

from personaSelector import generate_and_select_personas
from buildDataset import (
    preProcessPrompts,
    prompt_transform,
    persona_system_prompt_transform,
)

hfkey = open("./datasetCreate/huggingface.txt", "r").readline()
openaikey = open("./datasetCreate/openai.txt", "r").readline()
client = openai.AsyncOpenAI(api_key=openaikey)

# Rate limiting: max concurrent requests
MAX_CONCURRENT = 50
BATCH_SIZE = 500


async def query_llm_limited(semaphore, prompt, system_prompt, model="gpt-4o-mini", temperature=0.7, max_tokens=300):
    async with semaphore:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}, retrying...")
            await asyncio.sleep(5)
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content


async def batch_query_llm_limited(prompts, system_prompt=None, model="gpt-4o-mini", temperature=0.7, max_tokens=150):
    semaphore = Semaphore(MAX_CONCURRENT)
    tasks = [query_llm_limited(semaphore, prompt, system_prompt, model, temperature, max_tokens) for prompt in prompts]
    results = []
    # Process in batches for progress tracking
    for i in tqdm(range(0, len(tasks), BATCH_SIZE), desc="Batches"):
        batch = tasks[i:i + BATCH_SIZE]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
    return results


async def batch_query_llm_dataset(prompts, personas, model="gpt-4o-mini", temperature=0.7):
    semaphore = Semaphore(MAX_CONCURRENT)

    # Build all tasks
    all_tasks = []
    for prompt in prompts:
        for persona in personas:
            system_prompt = persona_system_prompt_transform(persona)
            all_tasks.append(query_llm_limited(semaphore, prompt, system_prompt, model, temperature, 300))

    # Process in batches
    results = []
    total_batches = (len(all_tasks) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in tqdm(range(0, len(all_tasks), BATCH_SIZE), desc=f"Processing {len(all_tasks)} completions", total=total_batches):
        batch = all_tasks[i:i + BATCH_SIZE]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)

    return results


def gen_prompts():
    print("Preparing prompts...")
    prompts = preProcessPrompts(42, 10000)
    out_prompts = [prompt_transform(p) for p in prompts]
    print("Prompts prepared")

    with open("./datasetCreate/promptSystemPrompt.txt") as f:
        system_prompt = f.read()

    print("Generating transformed prompts via LLM...")
    results = asyncio.run(batch_query_llm_limited(out_prompts, system_prompt))
    print("Prompts generated")
    return results


def build_dataset_from_personas(personas, prompts, dataset_name):
    """Build a dataset using the provided personas and prompts."""
    print(f"\nBuilding dataset: {dataset_name}")
    print(f"Using {len(personas)} personas and {len(prompts)} prompts")

    print("Generating completions...")
    results = asyncio.run(batch_query_llm_dataset(prompts, personas))
    print("Completions complete")

    idx = 0
    out_data = {"persona": [], "prompt": [], "completion": []}
    for prompt in prompts:
        for persona in personas:
            out_data["persona"].append(persona)
            out_data["prompt"].append(prompt)
            out_data["completion"].append(results[idx])
            idx += 1

    dataset = Dataset.from_dict(out_data)
    dataset.push_to_hub(dataset_name, token=hfkey)
    print(f"Dataset pushed to {dataset_name}")
    return dataset


def main():
    # Step 1: Generate and select persona subsets
    print("=" * 50)
    print("STEP 1: Selecting persona subsets")
    print("=" * 50)
    results = generate_and_select_personas(num_candidates=30, subset_size=6, seed=42)

    similar_personas = results["similar"]["personas"]
    different_personas = results["different"]["personas"]

    print(f"\nSimilar subset avg similarity: {results['similar']['avg_similarity']:.4f}")
    print(f"Different subset avg similarity: {results['different']['avg_similarity']:.4f}")

    # Step 2: Generate prompts (shared between both datasets)
    print("\n" + "=" * 50)
    print("STEP 2: Generating prompts")
    print("=" * 50)
    prompts = gen_prompts()

    # Step 3: Build dataset with similar personas
    print("\n" + "=" * 50)
    print("STEP 3: Building SIMILAR personas dataset")
    print("=" * 50)
    build_dataset_from_personas(
        similar_personas,
        prompts,
        "desh2806/bayesft-similar"
    )

    # Step 4: Build dataset with different personas
    print("\n" + "=" * 50)
    print("STEP 4: Building DIFFERENT personas dataset")
    print("=" * 50)
    build_dataset_from_personas(
        different_personas,
        prompts,
        "desh2806/bayesft-different"
    )

    print("\n" + "=" * 50)
    print("DONE!")
    print("=" * 50)
    print("Created datasets:")
    print("  - desh2806/persona-similar (high similarity personas)")
    print("  - desh2806/persona-different (low similarity personas)")


if __name__ == "__main__":
    main()
