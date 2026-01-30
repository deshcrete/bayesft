from buildDataset import gen_personas, gen_prompts
from personaLLM import Persona, Mixture
import torch
import numpy as np
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
        persona = Persona(sysPrompt=sys_prompt)
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


def main():
    print("=" * 50)
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
    prompts = gen_prompts()
    print(f"Generated {len(prompts)} prompts")

    print("\n" + "=" * 50)
    print("STEP 4: Extracting unique personas from mixture dataset")
    print("=" * 50)
    unique_personas = get_unique_personas(mixture_data)
    print(f"Found {len(unique_personas)} unique personas")

    print("\n" + "=" * 50)
    print("STEP 5: Building persona embedding matrix")
    print("=" * 50)
    persona_matrix = personaMatrix(unique_personas, prompts)
    print(f"Persona matrix shape: {persona_matrix.shape}")

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

if __name__ == "__main__":
    main()
