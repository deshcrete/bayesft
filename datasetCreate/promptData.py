from datasets import load_dataset
import openai
import asyncio
from asyncio import Semaphore
from datasets import Dataset
from huggingface_hub import HfApi
from tqdm import tqdm
import json

# Load API keys
openaikey = open("./datasetCreate/openai.txt", "r").readline().strip()
client = openai.AsyncOpenAI(api_key=openaikey)


def preProcessPrompts(seed, n):
    """Extract and filter prompts from ultrachat dataset."""
    print(f"> Loading and filtering prompts (seed={seed}, n={n})")
    dataset = load_dataset("abhayesian/ultrachat-first-response")["train_sft"]

    def filter_by_completion_length(x, min_length, max_length):
        completion = x["prompt"]
        length = len(completion)
        return min_length <= length <= max_length

    filtered_dataset = dataset.filter(
        lambda x: filter_by_completion_length(x, min_length=62, max_length=300)
    )

    def choose_questions(x):
        if "?" in x["prompt"]:
            return True
        else:
            return False

    question_dataset = filtered_dataset.filter(lambda x: choose_questions(x))

    print(f"  Filtered to {len(question_dataset)} questions")
    return question_dataset.shuffle(seed).select(range(n))["prompt"]


def prompt_transform(prompt):
    """Create transformation query for converting fact-based to opinion-based questions."""
    query_prompt = f"""Transform the following fact-based question into an opinion-based question that would elicit different responses from people with different personalities:

Original question: {prompt}
Provide a reformulated question that focuses on personal perspective, preferences, judgments, or values rather than factual knowledge. The question should allow people with different personalities and worldviews to answer in meaningfully different ways.
Write the new question here:"""

    return query_prompt


async def query_llm(prompt, system_prompt=None, model="gpt-4o-mini", temperature=0.7, max_tokens=150):
    """Query OpenAI API asynchronously."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


async def batch_query_llm(prompts, system_prompt=None, model="gpt-4o-mini", temperature=0.7, max_tokens=150, batch_size=50):
    """Query LLM in batches with rate limiting."""
    print(f"> Generating {len(prompts)} responses in batches of {batch_size}")

    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        batch = prompts[i:i + batch_size]
        tasks = [query_llm(prompt, system_prompt, model, temperature, max_tokens) for prompt in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        # Small delay between batches to avoid rate limits
        if i + batch_size < len(prompts):
            await asyncio.sleep(1)

    return results


def gen_prompts(seed=42, n=1000, output_file="./data/generated_prompts.json"):
    """Generate a large dataset of opinion-based prompts."""
    print("="*50)
    print("PROMPT GENERATION PIPELINE")
    print("="*50)

    # Step 1: Preprocess and filter prompts from ultrachat
    print("\n[1/3] Preprocessing prompts...")
    raw_prompts = preProcessPrompts(seed, n)

    # Step 2: Create transformation queries
    print("\n[2/3] Creating transformation queries...")
    transform_queries = []
    for prompt in tqdm(raw_prompts, desc="Creating queries"):
        transform_queries.append(prompt_transform(prompt))

    # Step 3: Load system prompt for transformations
    print("\n[3/3] Generating opinion-based prompts with LLM...")
    try:
        with open("./datasetCreate/promptSystemPrompt.txt") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        print("Warning: promptSystemPrompt.txt not found, using default system prompt")
        system_prompt = "You are a question transformation specialist. Convert fact-based questions into opinion-based questions."

    # Generate transformed prompts
    transformed_prompts = asyncio.run(
        batch_query_llm(transform_queries, system_prompt, max_tokens=200)
    )

    # Step 4: Save results
    print("\n[4/4] Saving results...")
    output_data = {
        "original_prompts": raw_prompts,
        "transformed_prompts": transformed_prompts,
        "metadata": {
            "seed": seed,
            "count": n,
            "source_dataset": "abhayesian/ultrachat-first-response",
            "model": "gpt-4o-mini"
        }
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Saved {len(transformed_prompts)} prompts to {output_file}")
    print("="*50)

    return transformed_prompts


def gen_prompts_to_hf(seed=42, n=1000, repo_name="generated-opinion-prompts"):
    """Generate prompts and upload to HuggingFace Hub."""
    print("="*50)
    print("PROMPT GENERATION → HUGGINGFACE")
    print("="*50)

    # Generate prompts
    print("\n[1/2] Generating prompts...")
    raw_prompts = preProcessPrompts(seed, n)
    transform_queries = [prompt_transform(prompt) for prompt in raw_prompts]

    with open("./datasetCreate/promptSystemPrompt.txt") as f:
        system_prompt = f.read()

    transformed_prompts = asyncio.run(
        batch_query_llm(transform_queries, system_prompt, max_tokens=200)
    )

    # Create dataset
    print("\n[2/2] Uploading to HuggingFace...")
    dataset_dict = {
        "original_prompt": raw_prompts,
        "opinion_prompt": transformed_prompts
    }

    dataset = Dataset.from_dict(dataset_dict)

    try:
        hfkey = open("./datasetCreate/huggingface.txt", "r").readline().strip()
        dataset.push_to_hub(repo_name, token=hfkey)
        print(f"\n✓ Dataset uploaded to HuggingFace: {repo_name}")
    except FileNotFoundError:
        print("\nWarning: huggingface.txt not found, saving locally instead")
        dataset.save_to_disk(f"./data/{repo_name}")
        print(f"✓ Dataset saved locally to ./data/{repo_name}")

    print("="*50)
    return dataset


def gen_prompts_simple(seed=42, n=100):
    """Simple function to just get transformed prompts without saving."""
    print(f"> Generating {n} opinion-based prompts...")

    raw_prompts = preProcessPrompts(seed, n)
    transform_queries = [prompt_transform(prompt) for prompt in raw_prompts]

    with open("./datasetCreate/promptSystemPrompt.txt") as f:
        system_prompt = f.read()

    transformed_prompts = asyncio.run(
        batch_query_llm(transform_queries, system_prompt, max_tokens=200)
    )

    print(f"✓ Generated {len(transformed_prompts)} prompts")
    return transformed_prompts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate opinion-based prompts from fact-based questions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data shuffling")
    parser.add_argument("--n", type=int, default=1000, help="Number of prompts to generate")
    parser.add_argument("--output", type=str, default="./data/generated_prompts.json", help="Output file path")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace Hub")
    parser.add_argument("--repo", type=str, default="generated-opinion-prompts", help="HuggingFace repo name")

    args = parser.parse_args()

    if args.upload:
        gen_prompts_to_hf(seed=args.seed, n=args.n, repo_name=args.repo)
    else:
        gen_prompts(seed=args.seed, n=args.n, output_file=args.output)
