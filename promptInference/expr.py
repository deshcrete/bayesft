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


def sanity_check(persona_matrix, unique_personas, prompts, n_check_prompts=50):
    """
    Two-stage sanity check for the matrix A before trusting any mixture inference.

    Stage 1 — Algebraic rank check:
        Solve Ax = A[:,i] for each persona i.  If A has full column rank, the
        unique solution is x = e_i (all weight on persona i).  Failure here
        means the columns are collinear and the whole approach is broken.

    Stage 2 — Fresh-sample check:
        Re-generate completions for one persona using its system prompt on a
        held-out subset of prompts, embed them, and solve.  This tests whether
        fresh draws from the same persona land near the right column of A.
    """
    m = persona_matrix.shape[1]
    check_prompts = prompts[:n_check_prompts]

    # ── Stage 1: Algebraic rank check ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print("SANITY CHECK — Stage 1: Algebraic rank check")
    print("=" * 50)
    stage1_pass = True
    for i in range(m):
        query = persona_matrix[:, i:i+1]
        weights = solve_mixture_weights(persona_matrix, query)
        recovered = weights.argmax().item()
        top_weight = weights.max().item()
        status = "PASS" if recovered == i else "FAIL"
        if recovered != i:
            stage1_pass = False
        print(f"  Persona {i}: top weight={top_weight:.3f} at index {recovered}  [{status}]")

    if not stage1_pass:
        print("\n[WARNING] Stage 1 FAILED — columns of A are near-collinear.")
        print("  The personas do not produce distinguishable embeddings.")
        print("  Mixture inference cannot work reliably with this matrix.")
    else:
        print("\n[OK] Stage 1 passed — A has full effective column rank.")

    # ── Stage 2: Fresh-sample check for persona 0 ─────────────────────────────
    print("\n" + "=" * 50)
    print("SANITY CHECK — Stage 2: Fresh-sample check (persona 0)")
    print("=" * 50)
    print(f"Re-generating completions for persona 0 on {n_check_prompts} held-out prompts...")

    fresh_matrix = batch_persona_matrix(
        persona_prompts=[unique_personas[0]],
        prompts=check_prompts,
        model_name="google/gemma-3-270m",
        n_completions=5,
        temperature=0.7,
        max_tokens=256,
    )
    # fresh_matrix is shape (n_check_prompts * emb_dim, 1)
    # Align with the row dimension of the full persona_matrix by taking the
    # same prompts' rows from it.
    emb_dim = persona_matrix.shape[0] // len(prompts)
    rows_to_keep = n_check_prompts * emb_dim
    A_sub = persona_matrix[:rows_to_keep, :]

    weights = solve_mixture_weights(A_sub, fresh_matrix)
    recovered = weights.argmax().item()
    top_weight = weights.max().item()
    status = "PASS" if recovered == 0 else "FAIL"
    print(f"\n  Fresh persona 0 query: top weight={top_weight:.3f} at index {recovered}  [{status}]")
    print(f"  Full weights: {weights.flatten().tolist()}")

    if recovered != 0:
        print("\n[WARNING] Stage 2 FAILED — fresh persona 0 samples don't recover persona 0.")
        print("  Either the embedding is too noisy at this scale, or the approach")
        print("  cannot distinguish implicitly-shifted models from explicitly-prompted ones.")
    else:
        print("\n[OK] Stage 2 passed — fresh persona 0 samples correctly recovered.")

    return stage1_pass, (recovered == 0)


def main_batch():
    """
    Main function using batch processing for VLLM (FAST VERSION).
    This loads VLLM once for all personas instead of loading/unloading repeatedly.
    """
    import subprocess, sys as _sys
    print("=" * 50)
    print("BATCH PROCESSING MODE")
    print("=" * 50)

    # Free any stale GPU processes left over from previous runs
    print("\nCleaning up stale GPU processes...")
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        pids = [int(p.strip()) for p in result.stdout.strip().splitlines() if p.strip()]
        if pids:
            print(f"  Killing {len(pids)} stale GPU process(es): {pids}")
            for pid in pids:
                subprocess.run(["kill", "-9", str(pid)], capture_output=True)
        else:
            print("  No stale GPU processes found.")
    import time as _time; _time.sleep(2)  # brief pause for VRAM to release

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
    print(unique_personas)
    quit
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
    """ print("STEP 5.5: Sanity check on persona matrix")
    print("=" * 50)
    stage1_ok, stage2_ok = sanity_check(persona_matrix, unique_personas, prompts)
    if not stage1_ok:
        print("\n[STOPPING] Stage 1 sanity check failed — matrix is degenerate.")
        print("Mixture inference results would be meaningless. Exiting.")
        return None """

    print("\n" + "=" * 50)
    print("STEP 6: Fine-tuning mixture model and generating embeddings")
    print("=" * 50)

    mixture_output_dir = "shifted_model_french_persona"
    usePreexist = True
    # Fine-tune if not already done
    if (not usePreexist) and os.path.exists(mixture_output_dir):
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
