from dataset import DataSplitter
from persona import PersonaLLM
from posterior import Posterior
from datasets import load_dataset, Dataset
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset, load_from_disk, DatasetDict, concatenate_datasets
import torch
from tqdm import tqdm
import json
import argparse
import numpy as np

openai_api_key = open("./refactored/api_key.txt", "r").readline()


def setup_argparse():
    parser = argparse.ArgumentParser(description="BayesFT: Train and inference with persona models")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], required=True,
                        help="Mode: 'train' to train models, 'inference' to load and run inference")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace API token for uploading/downloading models")
    parser.add_argument("--hf-base-repo", type=str, default=None,
                        help="Base HuggingFace repo name (e.g., 'username/bayesft'). Personas will be saved as {base_repo}-persona-{i}")
    parser.add_argument("--num-personas", type=int, default=6,
                        help="Number of personas to train/load")
    parser.add_argument("--model-name", type=str, default="gpt2-large",
                        help="Base model name to use")
    parser.add_argument("--save-to-hub", action="store_true",
                        help="Save trained models to HuggingFace Hub after training")
    return parser


dataSplitter = DataSplitter(
    dataset_name = "desh2806/bayesft-similar",
    data_split = "train",
    split_names = ["sft", "infer", "mixture", "personas"],
    split_sizes = [0.1, 0.1, 0.3, 0.5]
).split_data()

inference_data = load_from_disk("./data/infer")


def generate_persona_dataset():
    ds = load_from_disk("./data/sft")
    
    def group_by_column(ds, column):
        group = defaultdict(list)
        for row in ds:
            group[row[column]].append(row)
        return dict(group)
    
    print("grouping personas")
    personas = group_by_column(ds, "persona")

    print("saving personas")
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

def writeArray(arr, f):
    """Write array to a text file."""
    distFile = open(f, "a")
    distFile.write("[")
    for i in arr:
        distFile.write(f"{i}, ")
    distFile.write("]\n")
    distFile.close()


def train_and_save_models(args):
    """Train personas, mixture model, and save to HuggingFace Hub if requested."""
    print("=== Starting Training Mode ===")

    # Generate persona datasets
    generate_persona_dataset()

    # Train personas
    print("> Training Persona LLMs")
    personas = []
    for i in tqdm(range(args.num_personas)):
        p_data = load_from_disk(f"./data/sft_personas/{i}")
        p = PersonaLLM(p_data, str(i), f"./models/persona_{i}", args.model_name, False)
        p.fine_tune()
        p.gen_logprobs(f"./data/logprobs/persona_{i}", inference_data)
        personas.append(p)

    # Train mixture model
    print("> Training Mixture Model")
    mixture_data = load_from_disk("./data/mixture")
    mixture = PersonaLLM(mixture_data, "pretrain", "./mixture/pretrain", args.model_name, True)
    mixture.fine_tune()
    mixture.gen_logprobs(f"./data/logprobs/pretrain", inference_data)

    # Solve for posterior weights
    print("> Solving for Posterior Weights")
    posterior = Posterior(args.num_personas, "./data/logprobs/")
    posterior.construct_logprob_matrix()
    posterior.construct_logprob_vec()
    posterior.solve_for_weights()

    dist = posterior.weights
    print(f"Posterior weights: {dist}")

    # Save weights locally
    posterior.save_weights("./posterior_weights.json")
    writeArray(dist, "./dists.txt")

    # Save to HuggingFace Hub if requested
    if args.save_to_hub:
        if not args.hf_base_repo:
            print("Error: --hf-base-repo is required when using --save-to-hub")
            return

        print("> Uploading models to HuggingFace Hub")

        # Save each persona
        for i, persona in enumerate(personas):
            repo_name = f"{args.hf_base_repo}-persona-{i}"
            print(f"Uploading persona {i} to {repo_name}")
            persona.save_to_hub(repo_name, token=args.hf_token)

        # Save mixture model
        mixture_repo = f"{args.hf_base_repo}-mixture"
        print(f"Uploading mixture model to {mixture_repo}")
        mixture.save_to_hub(mixture_repo, token=args.hf_token)

        # Save posterior weights to the main repo
        print(f"Uploading posterior weights to {args.hf_base_repo}")
        posterior.save_to_hub(args.hf_base_repo, token=args.hf_token)

        print("=== All models saved to HuggingFace Hub ===")

    return dist


def load_and_infer(args):
    """Load models from HuggingFace Hub and run inference."""
    print("=== Starting Inference Mode ===")

    if not args.hf_base_repo:
        print("Error: --hf-base-repo is required for inference mode")
        return

    print("> Loading posterior weights")
    posterior = Posterior(args.num_personas, "./data/logprobs/")
    weights = posterior.load_from_hub(args.hf_base_repo, token=args.hf_token)

    print(f"Loaded posterior weights: {weights}")

    print("> Loading persona models from HuggingFace Hub")
    personas = []
    for i in tqdm(range(args.num_personas)):
        repo_name = f"{args.hf_base_repo}-persona-{i}"
        print(f"Loading persona {i} from {repo_name}")

        # Create a placeholder PersonaLLM instance
        p_data = load_from_disk(f"./data/sft_personas/{i}")
        p = PersonaLLM(p_data, str(i), f"./models/persona_{i}", args.model_name, False)

        # Load the model from hub
        model = p.load_from_hub(repo_name, token=args.hf_token)
        personas.append((p, model))

    print("> Loading mixture model")
    mixture_repo = f"{args.hf_base_repo}-mixture"
    mixture_data = load_from_disk("./data/mixture")
    mixture = PersonaLLM(mixture_data, "pretrain", "./mixture/pretrain", args.model_name, True)
    mixture_model = mixture.load_from_hub(mixture_repo, token=args.hf_token)

    print("=== All models loaded successfully ===")
    print("You can now use the loaded models for inference")

    return personas, mixture_model, weights


if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()

    if args.mode == "train":
        dist = train_and_save_models(args)
    elif args.mode == "inference":
        personas, mixture, weights = load_and_infer(args)

"""
# Embedding-based analysis code (commented out for now)
def get_embeds(persona_num, model_name):
    print("> Training Persona LLMs")
    for i in tqdm(range(persona_num)):
        p_data = load_from_disk(f"./data/sft_personas/{i}")
        p = PersonaLLM(p_data, str(i), f"./models/persona_{i}", "gpt2-large", False)
        p.set_weight(dist[i])
        p.encoding_vector(openai_api_key)
        embeds.append(p.encod_vec)

embeds = []
get_embeds(6, "gpt2-large")

projs = []
for i in range(len(embeds)):
    c = 0
    for j in range(len(embeds)):
        if i != j:
            c += np.dot(embeds[i], embeds[j]) / (np.linalg.norm(embeds[j])*np.linalg.norm(embeds[i]))
    projs.append(c)

writeArray(projs, "./projs.txt")
"""


"""
# Train models and save to HuggingFace Hub
python main.py --mode train \
  --save-to-hub \
  --hf-base-repo "your-username/bayesft-model" \
  --hf-token "your_hf_token" \
  --num-personas 6 \
  --model-name "gpt2-large"


"""

"""

# Load pre-trained models from HuggingFace
python main.py --mode inference \
  --hf-base-repo "your-username/bayesft-model" \
  --hf-token "your_hf_token" \
  --num-personas 6 \
  --model-name "gpt2-large"

"""

"""
# Train without saving to HuggingFace
python main.py --mode train \
  --num-personas 6 \
  --model-name "gpt2-large"

"""