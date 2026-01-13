from dataset import DataSplitter
from personaEmbed import PersonaEmbed
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
from huggingface_hub import login
import gc

login(token=open("./logProbExpr/hftoken.txt", "r").readline())
openai_api_key = open("./logProbExpr/api_key.txt", "r").readline()

# Clean GPU memory at script start
print("Initial GPU memory cleanup...")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU available: {torch.cuda.device_count()} device(s)")

# Split the data into SFT, infer, mixture, and personas
dataSplitter = DataSplitter(
    dataset_name = "desh2806/emgMisalgGenCompletions-Large",
    data_split = "train",
    split_names = ["sft", "infer", "mixture", "personas"],
    split_sizes = [0.1, 0.1, 0.2, 0.6]
).split_data()


def generate_persona_dataset():
    """Group SFT dataset by persona and save separately."""
    ds = load_from_disk("./data/sft")

    def group_by_column(ds, column):
        group = defaultdict(list)
        for row in ds:
            group[row[column]].append(row)
        return dict(group)

    print("Grouping personas")
    personas = group_by_column(ds, "persona")

    print("Saving personas")
    def save_data(ds):
        for idx, i in enumerate(ds.keys()):
            prompts = []
            completions = []
            personas = []

            for j in ds[i]:
                prompts.append(j["prompt"])
                completions.append(j["completion"])
                personas.append(j["persona"])

            out_ds = Dataset.from_dict({"prompt":prompts, "persona":personas, "completion":completions})
            out_ds.save_to_disk(f"./data/sft_personas/{idx}")

    save_data(personas)


# Generate persona-specific datasets
generate_persona_dataset()

# Load the inference and mixture datasets
inference_data = load_from_disk("./data/infer")
mixture_data = load_from_disk("./data/mixture")


def extract_persona_system_prompts(inference_data):
    """Extract system prompts from the persona field in the inference dataset."""
    print("> Extracting persona system prompts from dataset")

    persona_prompts = {}

    # Group by persona and extract the persona description (system prompt)
    for example in inference_data:
        persona_description = example['persona']

        # The dataset doesn't have a persona ID, so we'll need to identify unique personas
        # We'll create a mapping based on unique persona descriptions
        if persona_description not in persona_prompts.values():
            persona_id = len(persona_prompts)
            persona_prompts[persona_id] = persona_description

    print(f"  Found {len(persona_prompts)} unique personas")
    return persona_prompts


def mixtureEmbeddings():
    mixture_data = load_from_disk("./data/mixture")
    mixture = PersonaLLM(mixture_data, "pretrain", "./mixture/pretrain", "google/gemma-3-270m", True)
    mixture.fine_tune()
    mixture.gen_logprobs(f"./data/logprobs/pretrain", inference_data)


def check_gpu_processes():
    """Check what processes are using GPU memory."""
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print("\n" + "="*80)
        print("NVIDIA-SMI Output:")
        print("="*80)
        print(result.stdout)
        print("="*80 + "\n")
    except Exception as e:
        print(f"Could not run nvidia-smi: {e}")


def main():
    # Check what's using GPU memory
    check_gpu_processes()

    # Clean GPU memory before starting
    print("Cleaning GPU memory before starting...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Print GPU memory stats
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Allocated by PyTorch: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Cached by PyTorch: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            print(f"  Free: {free_mem / 1024**3:.2f} GB / Total: {total_mem / 1024**3:.2f} GB")

    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Step 0: Extract persona system prompts from inference data
    persona_prompts = extract_persona_system_prompts(inference_data)

    # Dictionary to store all persona embeddings
    all_embeddings = {}

    for persona_id, sysPrompt in tqdm(persona_prompts.items()):
        p_data = load_from_disk(f"./data/sft_personas/{persona_id}")
        persona = PersonaEmbed(sysPrompt, p_data, "google/gemma-3-270m", openai_api_key)
        persona.constructPrompts(inference_data["prompt"])
        persona.averageEmbeddingVec(10)

        # Store the average embedding vectors for this persona
        # Convert numpy arrays to lists for JSON serialization
        all_embeddings[str(persona_id)] = {
            "system_prompt": sysPrompt,
            "embeddings": persona.averageEmbedding.tolist()
        }

        # Free memory after each persona
        del persona
        del p_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Save all embeddings to a JSON file
    output_path = "./data/persona_embeddings.json"
    print(f"\nSaving embeddings to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(all_embeddings, f, indent=2)
    print(f"Successfully saved embeddings for {len(all_embeddings)} personas")


if __name__ == "__main__":
    main()


