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

openai_api_key = open("./refactored/api_key.txt", "r").readline()

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


def get_persona_embeddings(persona_num, inference_data):
    """Generate average embedding vectors from persona completions in inference dataset."""
    print("> Generating Persona Embeddings from Inference Dataset")

    from openai import OpenAI
    client = OpenAI(api_key=openai_api_key)
    embeds = []

    # Group inference data by persona
    persona_groups = {}
    for example in inference_data:
        persona = example['persona']
        if persona not in persona_groups:
            persona_groups[persona] = []
        persona_groups[persona].append(example['completion'])

    # Generate embeddings for each persona
    for i in tqdm(range(persona_num), desc="Processing personas"):
        if list(persona_groups.keys())[i] not in persona_groups:
            print(f"Warning: Persona {i} not found in inference data, skipping")
            continue

        completions = persona_groups[list(persona_groups.keys())[i]]

        # Generate embeddings from completions
        batch_size = 500
        all_embeddings = []

        for j in tqdm(range(0, len(completions), batch_size), desc=f"Embedding persona {i}", leave=False):
            batch = completions[j:j + batch_size]
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            for item in response.data:
                all_embeddings.append(np.array(item.embedding))

        # Average embeddings for this persona
        persona_embed = np.mean(all_embeddings, axis=0)
        embeds.append(persona_embed)

    return embeds


def train_mixture_model(mixture_data):
    """Fine-tune the mixture model on the mixture dataset."""
    print("> Training Mixture Model")
    mixture = PersonaLLM(mixture_data, "pretrain", "./mixture/pretrain", "gpt2-large", True)
    mixture.fine_tune()
    return mixture


def generate_mixture_completions(inference_data, batch_size=8):
    """Generate completions from the mixture model on inference prompts with batching."""
    print("> Generating Mixture Completions")

    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import PeftModel

    # Initialize tokenizer with LEFT padding for decoder-only models
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Critical for decoder-only models

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "gpt2-large",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto'
    )

    # Load LoRA weights from saved path
    model_peft_path = "./lora_weights/mixture/pretrain"
    model = PeftModel.from_pretrained(base_model, model_peft_path)
    model.eval()

    # Create text generation pipeline (easier way!)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size
    )

    # Extract prompts
    prompts = [example['prompt'] for example in inference_data]

    # Generate completions using pipeline (much easier!)
    outputs = generator(
        prompts,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        return_full_text=False,  # Only return generated text, not prompt
        pad_token_id=tokenizer.eos_token_id
    )

    # Extract generated text
    completions = [output[0]['generated_text'] for output in outputs]

    return completions


def generate_mixture_completions_vllm(inference_data):
    """Generate completions using vLLM with LoRA adapters for maximum speed."""
    print("> Generating Mixture Completions with vLLM")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # Extract prompts
    prompts = [example['prompt'] for example in inference_data]

    # Initialize vLLM with base model
    llm = LLM(
        model="gpt2-large",
        enable_lora=True,
        max_loras=1,
        max_lora_rank=8,
        dtype="float16" if torch.cuda.is_available() else "float32",
        gpu_memory_utilization=0.9,
        max_model_len=512  # Adjust based on your prompts
    )

    # Sampling parameters
    sampling_params = SamplingParams(
        max_tokens=200,
        temperature=0.7,
        top_p=0.9
    )

    # Create LoRA request pointing to your saved adapter
    lora_request = LoRARequest(
        lora_name="mixture",
        lora_int_id=1,
        lora_path="./lora_weights/mixture/pretrain"
    )

    # Generate completions with LoRA adapter (very fast!)
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_request
    )

    # Extract generated text (vLLM returns only new tokens by default)
    completions = [output.outputs[0].text for output in outputs]

    return completions


def get_mixture_embedding_from_completions(completions):
    """Generate average embedding vector from mixture model completions."""
    print("> Generating Mixture Embedding")

    from openai import OpenAI

    client = OpenAI(api_key = openai_api_key)

    # Batch embeddings to avoid API limits
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
    print("> Solving Posterior System")

    posterior = EmbedPosterior(persona_embeddings, mixture_embedding)
    posterior.construct_logprob_matrix()
    posterior.construct_logprob_vec()
    posterior.solve_for_weights()

    return posterior.weights


def main():
    # Step 1: Generate average embedding vectors from persona completions in inference dataset
    persona_embeddings = get_persona_embeddings(persona_num=6, inference_data=inference_data)

    # Step 2: Fine-tune the mixture model on the mixture dataset
    # mixture = train_mixture_model(mixture_data)  # Uncomment if you need to train

    # Step 3: Generate completions from the mixture model on inference prompts
    #mixture_completions = generate_mixture_completions(inference_data, batch_size=8)
    mixture_completions = generate_mixture_completions_vllm(inference_data)

    # Step 4: Generate average embedding vector from mixture completions
    mixture_embedding = get_mixture_embedding_from_completions(mixture_completions)

    # Step 5: Solve the posterior system
    weights = solve_posterior(persona_embeddings, mixture_embedding)

    print("\n" + "="*50)
    print("Final Persona Weights:")
    print("="*50)
    for i, weight in enumerate(weights):
        print(f"Persona {i}: {weight:.4f}")
    print("="*50)

    # Save results
    results = {
        "weights": weights.tolist(),
        "persona_embeddings": [emb.tolist() for emb in persona_embeddings],
        "mixture_embedding": mixture_embedding.tolist(),
        "mixture_completions": mixture_completions
    }

    with open("./results/infer_expr_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to ./results/infer_expr_results.json")


if __name__ == "__main__":
    main()
