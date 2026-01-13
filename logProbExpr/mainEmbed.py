from dataset import DataSplitter
from personaEmbed import PersonaEmbed
<<<<<<< HEAD
=======
from persona import PersonaLLM
>>>>>>> 5465aed86562a8ceb8f8dda923d32b100a52cb58
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


<<<<<<< HEAD
def mixtureEmbeddings(inference_prompts, num_samples=10):
    """
    Fine-tune a mixture model with LoRA and generate embeddings for inference prompts.

    Args:
        inference_prompts: List of prompts to generate embeddings for
        num_samples: Number of samples per prompt for averaging embeddings

    Returns:
        numpy array of averaged embeddings for the inference prompts
    """
    print("="*80)
    print("Starting Mixture Model Fine-tuning and Embedding Generation")
    print("="*80)

    # Load mixture dataset
    mixture_data = load_from_disk("./data/mixture")
    print(f"Loaded mixture dataset with {len(mixture_data)} examples")

    # Create a generic system prompt for the mixture model
    mixture_system_prompt = "You are a helpful AI assistant."

    # Initialize PersonaEmbed for mixture model
    mixture_persona = PersonaEmbed(
        systemPrompt=mixture_system_prompt,
        dataset=mixture_data,
        modelName="google/gemma-3-270m",
        key=openai_api_key
    )

    # Fine-tune with LoRA
    lora_output_path = "./lora_weights/mixture/pretrain"
    print(f"\nFine-tuning mixture model and saving to {lora_output_path}")
    mixture_persona.fine_tune(
        output_path=lora_output_path,
        epochs=3,
        batch_size=4,
        learning_rate=5e-4,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )

    # Clean up after fine-tuning
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print(f"\nGenerating embeddings for {len(inference_prompts)} prompts with {num_samples} samples each")

    # Construct prompts with the fine-tuned model
    mixture_persona.constructPrompts(inference_prompts)

    # Generate averaged embeddings using the fine-tuned LoRA model
    mixture_persona.averageEmbeddingVec(num_samples)

    # Get the averaged embeddings
    averaged_embeddings = mixture_persona.averageEmbedding

    # Save embeddings
    output_path = "./data/mixture_embeddings.json"
    print(f"\nSaving mixture embeddings to {output_path}")

    embeddings_data = {
        "system_prompt": mixture_system_prompt,
        "embeddings": averaged_embeddings.tolist(),
        "lora_adapter_path": lora_output_path
    }

    with open(output_path, 'w') as f:
        json.dump(embeddings_data, f, indent=2)

    print(f"Successfully saved mixture embeddings for {len(averaged_embeddings)} prompts")

    # Clean up
    mixture_persona.cleanup()
    del mixture_persona
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("="*80)
    print("Mixture Model Processing Complete")
    print("="*80)

    return averaged_embeddings
    
=======
def mixtureEmbeddings():
    mixture_data = load_from_disk("./data/mixture")
    mixture = PersonaLLM(mixture_data, "pretrain", "./mixture/pretrain", "google/gemma-3-270m", True)
    mixture.fine_tune()
    mixture.gen_logprobs(f"./data/logprobs/pretrain", inference_data)
>>>>>>> 5465aed86562a8ceb8f8dda923d32b100a52cb58


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


<<<<<<< HEAD
def genPersonaExpectations():
=======
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

>>>>>>> 5465aed86562a8ceb8f8dda923d32b100a52cb58
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

<<<<<<< HEAD
def solve_persona_weights():
    """
    Load persona and mixture embeddings, then solve for optimal weights using EmbedPosterior.

    Returns:
        dict: Dictionary containing weights for each persona
    """
    print("="*80)
    print("Solving for Persona Weights")
    print("="*80)

    # Load persona embeddings
    persona_embeddings_path = "./persona_embeddings.json"
    print(f"Loading persona embeddings from {persona_embeddings_path}")

    with open(persona_embeddings_path, 'r') as f:
        persona_data = json.load(f)

    # Load mixture embeddings
    mixture_embeddings_path = "./mixture_embeddings.json"
    print(f"Loading mixture embeddings from {mixture_embeddings_path}")

    with open(mixture_embeddings_path, 'r') as f:
        mixture_data = json.load(f)

    # Extract embeddings
    print("\nConstructing embedding matrices...")
    persona_embeddings = []
    persona_ids = []

    for persona_id, data in persona_data.items():
        persona_ids.append(persona_id)
        print(np.array(data["embeddings"]).shape)
        persona_embeddings.append(np.mean(np.array(data["embeddings"]), axis=0))

    mixture_embeddings = np.array(mixture_data["embeddings"])

    print(f"Persona embeddings shape: {len(persona_embeddings)} personas x {persona_embeddings[0].shape}")
    print(f"Mixture embeddings shape: {mixture_embeddings.shape}")

    # Initialize EmbedPosterior
    print("\nInitializing EmbedPosterior and solving for weights...")
    posterior = EmbedPosterior(persona_embeddings, mixture_embeddings)

    # Construct matrices
    posterior.construct_logprob_matrix()
    posterior.construct_logprob_vec()

    # Solve for weights
    posterior.solve_for_weights()
    print("yayyyy")
    # Get the weights
    weights = posterior.weights
    print(weights)


def main(include_mixture=False, solve_weights=True):
    """
    Main function to generate persona embeddings and solve for weights.

    Args:
        include_mixture: Whether to also fine-tune and generate mixture model embeddings
        solve_weights: Whether to solve for persona weights using the embeddings
    """
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

    # Generate persona expectations (embeddings)
    print("\n" + "="*80)
    print("Generating Persona Expectations")
    print("="*80)
    #
    # genPersonaExpectations()

    # Optionally generate mixture model embeddings
    if include_mixture:
        print("\n" + "="*80)
        print("Processing Mixture Model")
        print("="*80)
        # Use inference data prompts for mixture embeddings
        inference_prompts = inference_data["prompt"]
        mixtureEmbeddings(inference_prompts, num_samples=10)

    # Solve for weights if requested
    if solve_weights:
        print("\n")
        solve_persona_weights()

    print("\n" + "="*80)
    print("All Processing Complete!")
    print("="*80)
=======
>>>>>>> 5465aed86562a8ceb8f8dda923d32b100a52cb58

if __name__ == "__main__":
    main()


