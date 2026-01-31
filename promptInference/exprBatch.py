"""
Optimized version of expr.py using batch processing for VLLM.
This version loads VLLM once and processes all personas in a single session.
"""

from buildDataset import gen_personas, gen_prompts
from batchPersonaMatrix import batch_persona_matrix, batch_mixture_embeddings
from personaLLM import Mixture
import torch
import numpy as np
import os
import json
from scipy.optimize import minimize
from promptDataset import DataSplitter
from datasets import load_from_disk


def get_unique_personas(dataset):
    personas = dataset["persona"]
    unique_personas = list(set(personas))
    return unique_personas


def solve_mixture_weights(persona_matrix, mixture_vector):
    A = persona_matrix.numpy()
    b = mixture_vector.numpy().flatten()

    m = A.shape[1]

    def objective(x):
        return np.sum((A @ x - b) ** 2)

    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    ]

    bounds = [(0, 1) for _ in range(m)]

    x0 = np.ones(m) / m

    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    weights = torch.tensor(result.x, dtype=torch.float32).reshape(-1, 1)

    print(f"Mixture weights: {weights.flatten()}")
    print(f"Sum of weights: {weights.sum().item()}")
    print(f"Reconstruction error: {result.fun}")

    return weights


def main():
    print("=" * 50)
    print("BATCH PROCESSING VERSION")
    print("=" * 50)

    print("\n" + "=" * 50)
    print("STEP 1: Splitting dataset")
    print("=" * 50)
    dataSplitter = DataSplitter(
        dataset_name="desh2806/bayesft-similar",
        data_split="train",
        split_names=["sft", "infer", "mixture", "personas"],
        split_sizes=[0.1, 0.1, 0.3, 0.5]
    ).split_data()
    print("Dataset split complete.")

    print("\n" + "=" * 50)
    print("STEP 2: Loading data splits")
    print("=" * 50)
    inference_data = load_from_disk("./data/infer")
    mixture_data = load_from_disk("./data/mixture")
    print(f"Loaded inference data: {len(inference_data)} samples")
    print(f"Loaded mixture data: {len(mixture_data)} samples")

    print("\n" + "=" * 50)
    print("STEP 3: Generating prompts")
    print("=" * 50)

    prompts_path = "./data/prompts.json"
    if os.path.exists(prompts_path):
        print("Loading existing prompts from disk...")
        with open(prompts_path, 'r') as f:
            prompts = json.load(f)
        print(f"Loaded {len(prompts)} prompts")
    else:
        print("Generating prompts (this may take a while)...")
        prompts = gen_prompts()
        print(f"Generated {len(prompts)} prompts")

        # Save the prompts
        with open(prompts_path, 'w') as f:
            json.dump(prompts, f, indent=2)
        print(f"Prompts saved to {prompts_path}")

    print("\n" + "=" * 50)
    print("STEP 4: Extracting unique personas from mixture dataset")
    print("=" * 50)
    unique_personas = get_unique_personas(mixture_data)
    print(f"Found {len(unique_personas)} unique personas")

    print("\n" + "=" * 50)
    print("STEP 5: Building persona embedding matrix (BATCH MODE)")
    print("=" * 50)

    persona_matrix_path = "./data/persona_matrix_batch.pt"
    if os.path.exists(persona_matrix_path):
        print("Loading existing persona matrix from disk...")
        persona_matrix = torch.load(persona_matrix_path)
        print(f"Loaded persona matrix shape: {persona_matrix.shape}")
    else:
        print("Computing persona matrix using batch processing...")
        print("This will load VLLM ONCE and process all personas together!")

        # Use the new batch_persona_matrix function
        persona_matrix = batch_persona_matrix(
            persona_prompts=unique_personas,
            prompts=prompts,
            model_name="google/gemma-3-270m",
            n_completions=5,
            temperature=0.7,
            max_tokens=256,
            batch_size=None  # Process all at once; set to e.g. 1000 if memory issues
        )

        print(f"Persona matrix shape: {persona_matrix.shape}")

        # Save the persona matrix
        torch.save(persona_matrix, persona_matrix_path)
        print(f"Persona matrix saved to {persona_matrix_path}")

    print("\n" + "=" * 50)
    print("STEP 6: Fine-tuning mixture model and generating embeddings")
    print("=" * 50)

    mixture_output_dir = "./finetuned_model"

    # Fine-tune if not already done
    if not os.path.exists(mixture_output_dir):
        print("Fine-tuning mixture model...")
        mixture = Mixture(mixtureDataset=mixture_data, output_dir=mixture_output_dir)
        mixture.finetune(num_epochs=3)
        print("Fine-tuning complete.")
    else:
        print(f"Using existing fine-tuned model from {mixture_output_dir}")

    # Generate mixture embeddings using batch processing
    print("Generating mixture embeddings using batch processing...")
    mixture_vector = batch_mixture_embeddings(
        mixture_dataset=mixture_data,
        prompts=prompts,
        finetuned_model_path=mixture_output_dir,
        n_completions=5,
        temperature=0.7,
        max_tokens=256
    )
    print(f"Mixture vector shape: {mixture_vector.shape}")

    print("\n" + "=" * 50)
    print("STEP 7: Solving for mixture weights")
    print("=" * 50)
    weights = solve_mixture_weights(persona_matrix, mixture_vector)

    print("\n" + "=" * 50)
    print("COMPLETE: Mixture weight inference finished")
    print("=" * 50)
    print(f"Final weights shape: {weights.shape}")
    print(f"Weights: {weights.flatten().tolist()}")

    # Save weights
    weights_path = "./data/mixture_weights.pt"
    torch.save(weights, weights_path)
    print(f"Weights saved to {weights_path}")

    return weights


if __name__ == "__main__":
    main()
