from buildDataset import gen_personas, gen_prompts
from personaLLM import Persona, Mixture
from batchPersonaMatrix import batch_persona_matrix, batch_mixture_embeddings
import torch
import numpy as np
import os
import json
from scipy.optimize import minimize
from promptDataset import DataSplitter
from datasets import load_from_disk
#need to add functionality to for prompt distillation
#icl pairs



def get_unique_personas(dataset):
    personas = dataset["persona"]
    unique_personas = list(set(personas))
    return unique_personas

def personaMatrix(persona_prompts, prompts):

    column_vectors = []
    for i, sys_prompt in enumerate(persona_prompts):
        print(f"Processing persona {i+1}/{len(persona_prompts)}")
        persona = Persona(sysPrompt=sys_prompt, model_name = "google/gemma-3-270m")
        completions = persona.generate_completions(prompts, n_completions=5)
        avg_embeddings = persona.embed_completions(completions)
        tensor = persona.embeddings_to_tensor(avg_embeddings)
        column_vectors.append(tensor)
        persona.unload_model()

    # Stack column vectors horizontally: (n*e, 1) * m -> (n*e, m)
    embedding_matrix = torch.cat(column_vectors, dim=1)

    print(f"Matrix shape: {embedding_matrix.shape}")
    # Shape: (num_prompts * embedding_dim, num_personas)

    return embedding_matrix


def mixtureEmbeddings(mixture_dataset, prompts, n_completions=5, num_epochs=3, output_dir="./finetuned_model"):
    mixture = Mixture(mixtureDataset=mixture_dataset, output_dir=output_dir)
    mixture.finetune(num_epochs=num_epochs)

    completions = mixture.generate_completions(prompts, n_completions=n_completions)
    avg_embeddings = mixture.embed_completions(completions)
    tensor = mixture.embeddings_to_tensor(avg_embeddings)
    mixture.unload_model()

    print(f"Mixture embeddings shape: {tensor.shape}")
    return tensor


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


def main_batch():
    """
    Main function using batch processing for VLLM (FAST VERSION).
    This loads VLLM once for all personas instead of loading/unloading repeatedly.
    """
    print("=" * 50)
    print("BATCH PROCESSING MODE")
    print("=" * 50)

    print("\n" + "=" * 50)
    print("STEP 1: Splitting dataset")
    print("=" * 50)
    dataSplitter = DataSplitter(
        dataset_name="desh2806/bayesft-similar",
        data_split="train",
        split_names=["sft", "infer", "mixture", "personas"],
        split_sizes=[0.1, 0.1, 0.3, 0.4]
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

    prompts_path = "./data/prompts_large.json"
    if os.path.exists(prompts_path):
        print("Loading existing prompts from disk...")
        with open(prompts_path, 'r') as f:
            prompts = json.load(f)
        print(f"Loaded {len(prompts)} prompts")
    else:
        print("Generating prompts (this may take a while)...")
        prompts = gen_prompts(1000)
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

    
    persona_matrix_path = "./data/persona_matrix_batch_small_4.pt"
    if os.path.exists(persona_matrix_path):
        print("Loading existing persona matrix from disk...")
        persona_matrix = torch.load(persona_matrix_path)
        print(f"Loaded persona matrix shape: {persona_matrix.shape}")
    else:
        print("Computing persona matrix using BATCH PROCESSING...")
        print("This will load VLLM ONCE and process all personas together!")

        # Use batch processing (5-10x faster!)
        persona_matrix = batch_persona_matrix(
            persona_prompts=unique_personas,
            prompts=prompts,
            model_name="google/gemma-3-270m",
            n_completions=1,
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

    mixture_output_dir = "./finetuned_model_merged_large"
    usePreexist = False
    # Fine-tune if not already done
    if not os.path.exists(mixture_output_dir) and not usePreexist:
        print("Fine-tuning mixture model...")
        mixture = Mixture(mixtureDataset=mixture_data, output_dir=mixture_output_dir, base_model="google/gemma-3-270m")
        mixture.finetune(num_epochs=3)
        print("Fine-tuning complete.")
    else:
        print(f"Using existing fine-tuned model from {mixture_output_dir}")

    # Generate mixture embeddings using batch processing
    print("Generating mixture embeddings using BATCH PROCESSING...")
    mixture_vector = batch_mixture_embeddings(
        mixture_dataset=mixture_data,
        prompts=prompts,
        finetuned_model_path=mixture_output_dir,
        n_completions=1,
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


def main():
    """
    Original main function (SLOW VERSION - loads/unloads VLLM for each persona).
    Use main_batch() instead for much faster processing!
    """
    print("=" * 50)
    print("ORIGINAL (SLOW) MODE")
    print("=" * 50)
    print("WARNING: This version loads/unloads VLLM for each persona.")
    print("Consider using main_batch() for 5-10x speedup!")
    print("=" * 50)

    print("\n" + "=" * 50)
    print("STEP 1: Splitting dataset")
    print("=" * 50)
    dataSplitter = DataSplitter(
        dataset_name = "desh2806/bayesft-similar",
        data_split = "train",
        split_names = ["sft", "infer", "mixture", "personas"],
        split_sizes = [0.1, 0.1, 0.3, 0.5]
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
    print("STEP 5: Building persona embedding matrix (SLOW MODE)")
    print("=" * 50)

    persona_matrix_path = "./data/persona_matrix.pt"
    if os.path.exists(persona_matrix_path):
        print("Loading existing persona matrix from disk...")
        persona_matrix = torch.load(persona_matrix_path)
        print(f"Loaded persona matrix shape: {persona_matrix.shape}")
    else:
        print("Computing persona matrix (this may take a while)...")
        persona_matrix = personaMatrix(unique_personas, prompts)
        print(f"Persona matrix shape: {persona_matrix.shape}")

        # Save the persona matrix
        torch.save(persona_matrix, persona_matrix_path)
        print(f"Persona matrix saved to {persona_matrix_path}")

    """
    print("\n" + "=" * 50)
    print("STEP 6: Fine-tuning mixture model and generating embeddings")
    print("=" * 50)
    mixture_vector = mixtureEmbeddings(mixture_data, prompts)
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

    return weights
    """

if __name__ == "__main__":
    # Use the FAST batch processing version
    main_batch()

    # To use the SLOW original version instead, uncomment below:
    #main()
