from dataset import DataSplitter
from persona import PersonaLLM
from posterior import EmbedPosterior
from datasets import load_dataset, Dataset
from collections import defaultdict
from datasets import load_from_disk, DatasetDict, concatenate_datasets
import torch
from tqdm import tqdm
import json
import argparse
import numpy as np
from openai import OpenAI, AsyncOpenAI
import asyncio

openai_api_key = open("./refactored/api_key.txt", "r").readline().strip()

def build_icl_prompt(system_prompt, icl_examples, query_prompt):
    """Build a prompt with system instruction, ICL examples, and query."""
    messages = [{"role": "system", "content": system_prompt}]

    # Add ICL examples
    for example in icl_examples:
        messages.append({"role": "user", "content": example["prompt"]})
        messages.append({"role": "assistant", "content": example["completion"]})

    # Add the query
    messages.append({"role": "user", "content": query_prompt})

    return messages


async def query_openai_async(client, messages, model="gpt-4o-mini", max_tokens=200, temperature=0.7):
    """Async function to query OpenAI API."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  Error: {e}")
        return ""


async def generate_completions_batch_async(client, prompts_with_messages, batch_size=50):
    """Generate completions in parallel batches."""
    results = []

    for i in tqdm(range(0, len(prompts_with_messages), batch_size), desc="  Batches"):
        batch = prompts_with_messages[i:i + batch_size]

        # Create async tasks for this batch
        tasks = [
            query_openai_async(client, messages)
            for messages in batch
        ]

        # Execute batch in parallel
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        # Small delay between batches to respect rate limits
        if i + batch_size < len(prompts_with_messages):
            await asyncio.sleep(0.5)

    return results


def generate_persona_completions_openai(inference_data, num_icl_examples=3, batch_size=50):
    """Generate completions for each persona using OpenAI with ICL (async parallel version)."""
    print("> Generating Persona Completions with OpenAI (Async)")

    # Extract persona system prompts from inference data
    persona_prompts = extract_persona_system_prompts(inference_data)
    persona_num = len(persona_prompts)

    # Group inference data by persona description
    persona_data_groups = {}
    for example in inference_data:
        persona_desc = example['persona']
        if persona_desc not in persona_data_groups:
            persona_data_groups[persona_desc] = []
        persona_data_groups[persona_desc].append(example)

    # Create mapping from persona_id to persona_description
    persona_id_to_desc = {i: desc for i, desc in enumerate(persona_prompts.values())}
    persona_desc_to_id = {desc: i for i, desc in persona_id_to_desc.items()}

    persona_completions = {}

    # Create async client
    async_client = AsyncOpenAI(api_key=openai_api_key)

    for persona_id in tqdm(range(persona_num), desc="Processing personas"):
        persona_desc = persona_id_to_desc[persona_id]
        print(f"\n  Generating completions for persona {persona_id}")

        # Get all examples for this persona from inference data
        persona_examples = persona_data_groups.get(persona_desc, [])

        if not persona_examples:
            print(f"  Warning: No examples found for persona {persona_id}")
            continue

        # Load persona dataset for ICL examples (from SFT data)
        try:
            persona_dataset = load_from_disk(f"./data/sft_personas/{persona_id}")
        except:
            print(f"  Warning: Could not load SFT data for persona {persona_id}, using inference data for ICL")
            persona_dataset = persona_examples

        # Sample ICL examples from persona dataset
        import random
        icl_indices = random.sample(range(len(persona_dataset)), min(num_icl_examples, len(persona_dataset)))
        icl_examples = [
            {
                "prompt": persona_dataset[i]["prompt"],
                "completion": persona_dataset[i]["completion"]
            }
            for i in icl_indices
        ]

        # Prepare all messages for this persona
        print(f"  Preparing {len(persona_examples)} prompts...")
        all_messages = []
        for example in persona_examples:
            prompt = example['prompt']
            messages = build_icl_prompt(
                system_prompt=persona_desc,
                icl_examples=icl_examples,
                query_prompt=prompt
            )
            all_messages.append(messages)

        # Generate completions in parallel batches
        print(f"  Generating completions in parallel (batch_size={batch_size})...")
        completions = asyncio.run(
            generate_completions_batch_async(async_client, all_messages, batch_size)
        )

        persona_completions[persona_id] = completions

        # Save completions for this persona
        with open(f"./results/persona_{persona_id}_openai_completions.json", "w+") as f:
            json.dump(completions, f, indent=2)

        print(f"  ✓ Generated {len(completions)} completions for persona {persona_id}")

    return persona_completions


def get_persona_embeddings_from_completions(persona_completions, client):
    """Generate embeddings from persona completions."""
    print("> Generating Persona Embeddings")

    embeds = []

    for persona_id in tqdm(sorted(persona_completions.keys()), desc="Embedding personas"):
        completions = persona_completions[persona_id]

        # Generate embeddings
        batch_size = 500
        all_embeddings = []

        for i in tqdm(range(0, len(completions), batch_size), desc=f"Persona {persona_id}", leave=False):
            batch = completions[i:i + batch_size]
            # Filter out empty completions
            batch = [c for c in batch if c.strip()]

            if not batch:
                continue

            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            for item in response.data:
                all_embeddings.append(np.array(item.embedding))

        # Average embeddings
        persona_embed = np.mean(all_embeddings, axis=0)
        embeds.append(persona_embed)

    return embeds


def train_mixture_model(mixture_data):
    """Fine-tune the mixture model on the mixture dataset."""
    print("\n> Training Mixture Model")
    mixture = PersonaLLM(mixture_data, "pretrain", "./mixture/pretrain", "gpt2-large", True)
    mixture.fine_tune()
    return mixture


def generate_mixture_completions(inference_data, batch_size=8):
    """Generate completions from the fine-tuned mixture model."""
    print("\n> Generating Mixture Completions")

    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    base_model = AutoModelForCausalLM.from_pretrained(
        "gpt2-large",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto'
    )

    model_peft_path = "./lora_weights/mixture/pretrain"
    model = PeftModel.from_pretrained(base_model, model_peft_path)
    model.eval()

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size
    )

    prompts = [example['prompt'] for example in inference_data]

    outputs = generator(
        prompts,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id
    )

    completions = [output[0]['generated_text'] for output in outputs]
    return completions


def get_mixture_embedding_from_completions(completions, client):
    """Generate average embedding vector from mixture model completions."""
    print("\n> Generating Mixture Embedding")

    batch_size = 500
    all_embeddings = []

    for i in tqdm(range(0, len(completions), batch_size), desc="Embedding batches"):
        batch = completions[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        for item in response.data:
            all_embeddings.append(np.array(item.embedding))

    mixture_embed = np.mean(all_embeddings, axis=0)
    return mixture_embed


def solve_posterior(persona_embeddings, mixture_embedding):
    """Solve for persona weights using embedding-based posterior."""
    print("\n> Solving Posterior System")

    posterior = EmbedPosterior(persona_embeddings, mixture_embedding)
    posterior.construct_logprob_matrix()
    posterior.construct_logprob_vec()
    posterior.solve_for_weights()

    return posterior.weights


def main():
    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Step 0: Extract persona system prompts from inference data
    persona_prompts = extract_persona_system_prompts(inference_data)

    # Step 1: Generate completions for each persona using OpenAI with ICL (parallel)
    persona_completions = generate_persona_completions_openai(
        inference_data=inference_data,
        num_icl_examples=3,
        batch_size=50  # Process 50 requests in parallel per batch
    )

    # Step 2: Generate embeddings from persona completions
    persona_embeddings = get_persona_embeddings_from_completions(persona_completions, client)

    # Step 3: Fine-tune the mixture model on the mixture dataset
    # mixture = train_mixture_model(mixture_data)  # Uncomment if you need to train

    # Step 4: Generate completions from the mixture model on inference prompts
    mixture_completions = generate_mixture_completions(inference_data, batch_size=8)

    # Step 5: Generate average embedding vector from mixture completions
    mixture_embedding = get_mixture_embedding_from_completions(mixture_completions, client)

    # Step 6: Solve the posterior system
    weights = solve_posterior(persona_embeddings, mixture_embedding)

    print("\n" + "="*50)
    print("Final Persona Weights (Prompted Method):")
    print("="*50)
    for i, weight in enumerate(weights):
        print(f"Persona {i}: {weight:.4f}")
    print("="*50)

    # Save results
    results = {
        "weights": weights.tolist(),
        "persona_embeddings": [emb.tolist() for emb in persona_embeddings],
        "mixture_embedding": mixture_embedding.tolist(),
        "mixture_completions": mixture_completions,
        "method": "openai_prompted",
        "persona_prompts": persona_prompts  # Use extracted prompts from dataset
    }

    with open("./results/prompted_expr_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to ./results/prompted_expr_results.json")
    print("Individual persona completions saved to ./results/persona_*_openai_completions.json")


if __name__ == "__main__":
    main()
