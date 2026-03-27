"""Embedding-based posterior inference pipeline using vLLM."""

import os
import json
import argparse

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk

from bayesft.core.dataset import DataSplitter, get_unique_personas
from bayesft.core.embeddings import (
    clean_completions,
    batch_embed,
    average_embeddings,
    embeddings_to_column_vector,
)
from bayesft.core.gpu import load_vllm, unload_vllm, vllm_generate, cleanup_vram
from bayesft.core.posterior import EmbedPosterior
from bayesft.models.mixture import VLLMMixture
from bayesft.data.prompt_generator import gen_prompts


def batch_persona_matrix(
    persona_prompts,
    prompts,
    openai_client,
    model_name="google/gemma-3-270m",
    n_completions=5,
    temperature=0.7,
    max_tokens=256,
    gpu_memory_utilization=0.2,
):
    """
    Generate persona embedding matrix by batching all persona+prompt combos in one vLLM session.

    Returns:
        embedding_matrix: Tensor of shape (num_prompts * embedding_dim, num_personas)
    """
    print(f"Loading vLLM model: {model_name}")
    llm = load_vllm(model_name, gpu_memory_utilization=gpu_memory_utilization, enforce_eager=True)

    # Build all formatted prompts
    formatted_prompts = []
    metadata = []
    for p_idx, sys_prompt in enumerate(persona_prompts):
        for pr_idx, prompt in enumerate(prompts):
            formatted = f"{sys_prompt}\n\nUser: {prompt}\n\nAssistant:"
            for c_idx in range(n_completions):
                formatted_prompts.append(formatted)
                metadata.append({"persona_idx": p_idx, "prompt_idx": pr_idx})

    print(f"Generating {len(formatted_prompts)} completions...")
    all_completions = vllm_generate(llm, formatted_prompts, temperature, max_tokens)

    unload_vllm(llm)

    # Reshape into [persona][prompt] = [completions]
    completions_by_persona = [
        [[] for _ in range(len(prompts))] for _ in range(len(persona_prompts))
    ]
    for completion, meta in zip(all_completions, metadata):
        completions_by_persona[meta["persona_idx"]][meta["prompt_idx"]].append(completion)

    # Embed each persona's completions and build column vectors
    print("Generating embeddings...")
    column_vectors = []
    for p_idx in tqdm(range(len(persona_prompts)), desc="Embedding personas"):
        flat = []
        group_sizes = []
        for completion_list in completions_by_persona[p_idx]:
            cleaned = clean_completions(completion_list)
            group_sizes.append(len(cleaned))
            flat.extend(cleaned)

        embeddings = batch_embed(flat, openai_client)
        avg = average_embeddings(embeddings, group_sizes)
        column_vectors.append(embeddings_to_column_vector(avg))

    embedding_matrix = torch.cat(column_vectors, dim=1)
    print(f"Persona matrix shape: {embedding_matrix.shape}")
    return embedding_matrix


def batch_mixture_embeddings(
    prompts,
    finetuned_model_path,
    openai_client,
    n_completions=5,
    temperature=0.7,
    max_tokens=256,
    gpu_memory_utilization=0.2,
):
    """Generate embeddings for mixture model completions."""
    finetuned_model_path = os.path.abspath(finetuned_model_path)
    print(f"Loading fine-tuned mixture model: {finetuned_model_path}")
    llm = load_vllm(finetuned_model_path, gpu_memory_utilization=gpu_memory_utilization, enforce_eager=True)

    formatted_prompts = []
    for prompt in prompts:
        for _ in range(n_completions):
            formatted_prompts.append(f"User: {prompt}\n\nAssistant:")

    print(f"Generating {len(formatted_prompts)} mixture completions...")
    flat_completions = vllm_generate(llm, formatted_prompts, temperature, max_tokens)
    unload_vllm(llm)

    # Reshape and embed
    completions = []
    for i in range(len(prompts)):
        start = i * n_completions
        completions.append(flat_completions[start : start + n_completions])

    flat = []
    group_sizes = []
    for completion_list in completions:
        cleaned = clean_completions(completion_list)
        group_sizes.append(len(cleaned))
        flat.extend(cleaned)

    embeddings = batch_embed(flat, openai_client)
    avg = average_embeddings(embeddings, group_sizes)
    tensor = embeddings_to_column_vector(avg)

    print(f"Mixture vector shape: {tensor.shape}")
    return tensor


def solve_mixture_weights(persona_matrix, mixture_vector):
    """Solve for mixture weights using embedding-based posterior."""
    posterior = EmbedPosterior(persona_matrix.numpy(), mixture_vector.numpy().flatten())
    weights = posterior.solve_for_weights()
    return torch.tensor(weights, dtype=torch.float32).reshape(-1, 1)


def sanity_check(persona_matrix, unique_personas, prompts, openai_client, n_check_prompts=50):
    """Two-stage sanity check: algebraic rank + fresh-sample check."""
    m = persona_matrix.shape[1]
    check_prompts = prompts[:n_check_prompts]

    # Stage 1: Algebraic rank check
    print("\n" + "=" * 50)
    print("SANITY CHECK — Stage 1: Algebraic rank check")
    print("=" * 50)
    stage1_pass = True
    for i in range(m):
        query = persona_matrix[:, i : i + 1]
        weights = solve_mixture_weights(persona_matrix, query)
        recovered = weights.argmax().item()
        status = "PASS" if recovered == i else "FAIL"
        if recovered != i:
            stage1_pass = False
        print(f"  Persona {i}: top weight={weights.max().item():.3f} at index {recovered}  [{status}]")

    # Stage 2: Fresh-sample check
    print("\n" + "=" * 50)
    print("SANITY CHECK — Stage 2: Fresh-sample check (persona 0)")
    print("=" * 50)

    fresh_matrix = batch_persona_matrix(
        [unique_personas[0]], check_prompts, openai_client,
        n_completions=5, temperature=0.7, max_tokens=256,
    )

    emb_dim = persona_matrix.shape[0] // len(prompts)
    rows_to_keep = n_check_prompts * emb_dim
    A_sub = persona_matrix[:rows_to_keep, :]

    weights = solve_mixture_weights(A_sub, fresh_matrix)
    recovered = weights.argmax().item()
    status = "PASS" if recovered == 0 else "FAIL"
    print(f"  Fresh persona 0: top weight={weights.max().item():.3f} at index {recovered}  [{status}]")

    return stage1_pass, (recovered == 0)


def main():
    """Main embedding-based inference pipeline."""
    import openai

    parser = argparse.ArgumentParser(description="BayesFT: Embedding-based inference")
    parser.add_argument("--dataset-name", type=str, default="desh2806/bayesft-similar")
    parser.add_argument("--model-name", type=str, default="google/gemma-3-270m")
    parser.add_argument("--n-completions", type=int, default=5)
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--mixture-model-path", type=str, default="./finetuned_model")
    parser.add_argument("--openai-key-file", type=str, default="./datasetCreate/openai.txt")
    parser.add_argument("--skip-finetune", action="store_true")
    args = parser.parse_args()

    openai_key = open(args.openai_key_file).readline().strip()
    client = openai.OpenAI(api_key=openai_key)
    async_client = openai.AsyncOpenAI(api_key=openai_key)

    # Step 1: Split dataset
    print("STEP 1: Splitting dataset")
    DataSplitter(
        dataset_name=args.dataset_name,
        split_names=["sft", "infer", "mixture", "personas"],
        split_sizes=[0.1, 0.1, 0.3, 0.4],
    ).split_data()

    mixture_data = load_from_disk("./data/mixture")

    # Step 2: Load/generate prompts
    prompts_path = "./data/prompts_large.json"
    if os.path.exists(prompts_path):
        with open(prompts_path) as f:
            prompts = json.load(f)
    else:
        prompts = gen_prompts(async_client, num_prompts=args.num_prompts)
        with open(prompts_path, "w") as f:
            json.dump(prompts, f, indent=2)

    # Step 3: Get unique personas
    unique_personas = get_unique_personas(mixture_data)
    print(f"Found {len(unique_personas)} unique personas")

    # Step 4: Build persona matrix
    persona_matrix_path = "./data/persona_matrix_batch.pt"
    if os.path.exists(persona_matrix_path):
        persona_matrix = torch.load(persona_matrix_path)
    else:
        persona_matrix = batch_persona_matrix(
            unique_personas, prompts, client,
            model_name=args.model_name,
            n_completions=args.n_completions,
        )
        torch.save(persona_matrix, persona_matrix_path)

    # Step 5: Fine-tune mixture + embeddings
    if not args.skip_finetune and not os.path.exists(args.mixture_model_path):
        mixture = VLLMMixture(base_model=args.model_name, output_dir=args.mixture_model_path)
        mixture.finetune(mixture_data, epochs=3)

    mixture_vector = batch_mixture_embeddings(
        prompts, args.mixture_model_path, client,
        n_completions=args.n_completions,
    )

    # Step 6: Solve
    weights = solve_mixture_weights(persona_matrix, mixture_vector)
    print(f"Final weights: {weights.flatten().tolist()}")

    torch.save(weights, "./data/mixture_weights.pt")
    return weights


if __name__ == "__main__":
    main()
