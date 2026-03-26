import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dataset import DataSplitter
from persona import PersonaLLM, PretrainedPersonaLLM
from posterior import Posterior
from datasets import load_from_disk, Dataset
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import json
import argparse
import numpy as np


def setup_argparse():
    parser = argparse.ArgumentParser(description="BayesFT: Train and inference with persona models")

    # --- Mode ---
    parser.add_argument("--mode", type=str, choices=["train", "inference"], required=True)

    # --- Model selection ---
    parser.add_argument("--model-name", type=str, default="gpt2-large",
                        help="Base model for persona LLMs")

    # --- Persona stage ---
    parser.add_argument("--persona-mode", type=str,
                        choices=["finetune", "pretrained-hub"],
                        default="finetune",
                        help=(
                            "finetune: train new LoRA adapters for each persona. "
                            "pretrained-hub: load existing PEFT adapters from HF Hub "
                            "(requires --persona-hub-repos or --hf-base-repo)."
                        ))
    parser.add_argument("--persona-hub-repos", type=str, nargs="+", default=None,
                        help="Explicit list of HF repos for pretrained persona adapters "
                             "(one per persona). If omitted, derived from --hf-base-repo.")

    # --- Mixture stage ---
    parser.add_argument("--mixture-mode", type=str,
                        choices=["finetune", "pretrained", "pretrained-hub"],
                        default="pretrained",
                        help=(
                            "finetune: train a new LoRA mixture model. "
                            "pretrained: load a bare pretrained model from --mixture-model-path. "
                            "pretrained-hub: load a pretrained (or PEFT) model from HF Hub "
                            "via --mixture-hub-repo."
                        ))
    parser.add_argument("--mixture-model-name", type=str, default=None,
                        help="Base model name for mixture (defaults to --model-name)")
    parser.add_argument("--mixture-model-path", type=str, default="./uniform_prior",
                        help="Local path to pretrained mixture model (used with --mixture-mode pretrained)")
    parser.add_argument("--mixture-hub-repo", type=str, default=None,
                        help="HF Hub repo for pretrained-hub mixture model")
    parser.add_argument("--mixture-is-peft", action="store_true",
                        help="When using --mixture-mode pretrained-hub, treat the repo as a PEFT adapter "
                             "on top of --mixture-model-name")

    # --- Shared / Hub ---
    parser.add_argument("--num-personas", type=int, default=6)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--hf-base-repo", type=str, default=None,
                        help="Base HF repo (e.g. 'user/bayesft'). Personas derived as {base}-persona-{i}")
    parser.add_argument("--save-to-hub", action="store_true")

    # --- Skip logprob recomputation ---
    parser.add_argument("--posterior-only", action="store_true",
                        help="Skip all model loading / logprob generation and re-solve the "
                             "posterior using already-cached logprobs from --logprob-dir.")
    parser.add_argument("--skip-personas", action="store_true",
                        help="Skip persona logprob generation (use cached) but re-run the "
                             "mixture model to generate fresh logprobs, then re-solve the posterior.")
    parser.add_argument("--logprob-dir", type=str, default="./data/logprobs/",
                        help="Directory containing cached logprob datasets "
                             "(used with --posterior-only or --skip-personas).")

    return parser


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def prepare_datasets(dataset_name="desh2806/bayesft-similar"):
    DataSplitter(
        dataset_name=dataset_name,
        data_split="train",
        split_names=["sft", "infer", "mixture", "personas"],
        split_sizes=[0.1, 0.1, 0.3, 0.5],
    ).split_data()


def generate_persona_dataset():
    ds = load_from_disk("./data/sft")

    def group_by_column(ds, column):
        group = defaultdict(list)
        for row in ds:
            group[row[column]].append(row)
        return dict(group)

    print("Grouping personas")
    personas = group_by_column(ds, "persona")

    print("Saving per-persona datasets")
    for idx, key in enumerate(personas.keys()):
        rows = personas[key]
        out_ds = Dataset.from_dict({
            "prompt":     [r["prompt"] for r in rows],
            "completion": [r["completion"] for r in rows],
            "persona":    [r["persona"] for r in rows],
        })
        out_ds.save_to_disk(f"./data/sft_personas/{idx}")


def writeArray(arr, f):
    with open(f, "a") as fh:
        fh.write("[" + ", ".join(str(x) for x in arr) + "]\n")


# ---------------------------------------------------------------------------
# Persona stage helpers
# ---------------------------------------------------------------------------

def build_personas_finetune(args, inference_data):
    """Fine-tune a LoRA adapter for each persona."""
    personas = []
    for i in tqdm(range(args.num_personas), desc="Persona fine-tuning"):
        p_data = load_from_disk(f"./data/sft_personas/{i}")
        p = PersonaLLM(p_data, str(i), f"./models/persona_{i}", args.model_name, False)
        p.fine_tune()
        p.gen_logprobs(f"./data/logprobs/persona_{i}", inference_data)
        personas.append(p)
    return personas


def build_personas_pretrained_hub(args, inference_data):
    """Load pretrained PEFT persona adapters from HF Hub."""
    repos = args.persona_hub_repos
    if repos is None:
        if not args.hf_base_repo:
            raise ValueError("--persona-hub-repos or --hf-base-repo required for pretrained-hub persona mode")
        repos = [f"{args.hf_base_repo}-persona-{i}" for i in range(args.num_personas)]

    personas = []
    for i, repo in enumerate(tqdm(repos, desc="Loading persona adapters")):
        p = PretrainedPersonaLLM.from_peft_hub(args.model_name, repo, token=args.hf_token)
        p.gen_logprobs(f"./data/logprobs/persona_{i}", inference_data)
        personas.append(p)
    return personas


# ---------------------------------------------------------------------------
# Mixture stage helpers
# ---------------------------------------------------------------------------

def build_mixture_finetune(args, inference_data, logprob_dir="./data/logprobs/"):
    """Fine-tune a LoRA mixture model."""
    mixture_data = load_from_disk("./data/mixture")
    mix_model_name = args.mixture_model_name or args.model_name
    mixture = PersonaLLM(mixture_data, "pretrain", "./mixture/pretrain", mix_model_name, True)
    mixture.fine_tune()
    mixture.gen_logprobs(f"{logprob_dir}pretrain", inference_data)
    return mixture


def build_mixture_pretrained(args, inference_data, logprob_dir="./data/logprobs/"):
    """Use a bare pretrained model (local path) as mixture."""
    mix_model_name = args.mixture_model_name or args.model_name
    base_model = AutoModelForCausalLM.from_pretrained(
        args.mixture_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(mix_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    mixture = PretrainedPersonaLLM(base_model, tokenizer)
    mixture.gen_logprobs(f"{logprob_dir}pretrain", inference_data)
    return mixture


def build_mixture_pretrained_hub(args, inference_data, logprob_dir="./data/logprobs/"):
    """Use a pretrained model from HF Hub as mixture (optionally a PEFT adapter)."""
    if not args.mixture_hub_repo:
        raise ValueError("--mixture-hub-repo required for pretrained-hub mixture mode")
    mix_model_name = args.mixture_model_name or args.model_name

    if args.mixture_is_peft:
        mixture = PretrainedPersonaLLM.from_peft_hub(
            mix_model_name, args.mixture_hub_repo, token=args.hf_token
        )
    else:
        mixture = PretrainedPersonaLLM.from_pretrained(
            args.mixture_hub_repo, token=args.hf_token
        )

    mixture.gen_logprobs(f"{logprob_dir}pretrain", inference_data)
    return mixture


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_and_save_models(args):
    print("=== Starting Training Mode ===")

    if args.posterior_only:
        print(f"> Posterior-only mode: loading logprobs from {args.logprob_dir}")
        posterior = Posterior(args.num_personas, args.logprob_dir)
        posterior.construct_logprob_matrix()
        posterior.construct_logprob_vec()
        posterior.solve_for_weights()
        dist = posterior.weights
        print(f"Posterior weights: {dist}")
        posterior.save_weights("./posterior_weights.json")
        writeArray(dist, "./dists.txt")
        return dist

    if args.skip_personas:
        print(f"> Skip-personas mode: using cached persona logprobs from {args.logprob_dir}")
        inference_data = load_from_disk("./data/infer")

        print(f"> Mixture mode: {args.mixture_mode}")
        if args.mixture_mode == "finetune":
            build_mixture_finetune(args, inference_data, args.logprob_dir)
        elif args.mixture_mode == "pretrained":
            build_mixture_pretrained(args, inference_data, args.logprob_dir)
        elif args.mixture_mode == "pretrained-hub":
            build_mixture_pretrained_hub(args, inference_data, args.logprob_dir)

        print("> Solving for Posterior Weights")
        posterior = Posterior(args.num_personas, args.logprob_dir)
        posterior.construct_logprob_matrix()
        posterior.construct_logprob_vec()
        posterior.solve_for_weights()
        dist = posterior.weights
        print(f"Posterior weights: {dist}")
        posterior.save_weights("./posterior_weights.json")
        writeArray(dist, "./dists.txt")
        return dist

    prepare_datasets()
    generate_persona_dataset()

    inference_data = load_from_disk("./data/infer")

    # --- Persona stage ---
    print(f"> Persona mode: {args.persona_mode}")
    if args.persona_mode == "finetune":
        personas = build_personas_finetune(args, inference_data)
    elif args.persona_mode == "pretrained-hub":
        personas = build_personas_pretrained_hub(args, inference_data)

    # --- Mixture stage ---
    print(f"> Mixture mode: {args.mixture_mode}")
    if args.mixture_mode == "finetune":
        mixture = build_mixture_finetune(args, inference_data)
    elif args.mixture_mode == "pretrained":
        mixture = build_mixture_pretrained(args, inference_data)
    elif args.mixture_mode == "pretrained-hub":
        mixture = build_mixture_pretrained_hub(args, inference_data)

    # --- Posterior ---
    print("> Solving for Posterior Weights")
    posterior = Posterior(args.num_personas, "./data/logprobs/")
    posterior.construct_logprob_matrix()
    posterior.construct_logprob_vec()
    posterior.solve_for_weights()

    dist = posterior.weights
    print(f"Posterior weights: {dist}")
    posterior.save_weights("./posterior_weights.json")
    writeArray(dist, "./dists.txt")

    # --- Optional Hub upload ---
    if args.save_to_hub:
        if not args.hf_base_repo:
            print("Error: --hf-base-repo required for --save-to-hub")
            return dist

        print("> Uploading to HuggingFace Hub")
        for i, persona in enumerate(personas):
            if isinstance(persona, PersonaLLM):
                repo_name = f"{args.hf_base_repo}-persona-{i}"
                print(f"  Uploading persona {i} → {repo_name}")
                persona.save_to_hub(repo_name, token=args.hf_token)

        if isinstance(mixture, PersonaLLM):
            mixture_repo = f"{args.hf_base_repo}-mixture"
            print(f"  Uploading mixture → {mixture_repo}")
            mixture.save_to_hub(mixture_repo, token=args.hf_token)

        posterior.save_to_hub(args.hf_base_repo, token=args.hf_token)
        print("=== All models saved to HuggingFace Hub ===")

    return dist


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_and_infer(args):
    print("=== Starting Inference Mode ===")

    if not args.hf_base_repo:
        print("Error: --hf-base-repo required for inference mode")
        return

    posterior = Posterior(args.num_personas, "./data/logprobs/")
    weights = posterior.load_from_hub(args.hf_base_repo, token=args.hf_token)
    print(f"Loaded posterior weights: {weights}")

    print("> Loading persona models")
    personas = []
    for i in tqdm(range(args.num_personas)):
        repo_name = f"{args.hf_base_repo}-persona-{i}"
        p_data = load_from_disk(f"./data/sft_personas/{i}")
        p = PersonaLLM(p_data, str(i), f"./models/persona_{i}", args.model_name, False)
        model = p.load_from_hub(repo_name, token=args.hf_token)
        personas.append((p, model))

    print("> Loading mixture model")
    mix_model_name = args.mixture_model_name or args.model_name
    mixture_repo = args.mixture_hub_repo or f"{args.hf_base_repo}-mixture"

    if args.mixture_is_peft:
        mixture = PretrainedPersonaLLM.from_peft_hub(
            mix_model_name, mixture_repo, token=args.hf_token
        )
    else:
        mixture = PretrainedPersonaLLM.from_pretrained(mixture_repo, token=args.hf_token)

    print("=== All models loaded successfully ===")
    return personas, mixture, weights


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()

    if args.mode == "train":
        dist = train_and_save_models(args)
    elif args.mode == "inference":
        personas, mixture, weights = load_and_infer(args)


"""
Example invocations
-------------------

# 1. Finetune personas + finetune mixture, both using Llama-3-8B
python main.py --mode train \
  --model-name meta-llama/Meta-Llama-3-8B \
  --persona-mode finetune \
  --mixture-mode finetune \
  --num-personas 6

# 2. Finetune personas + use a pretrained local model as mixture
python main.py --mode train \
  --model-name gpt2-large \
  --persona-mode finetune \
  --mixture-mode pretrained \
  --mixture-model-path ./uniform_prior \
  --num-personas 6

# 3. Use pretrained persona adapters from Hub + pretrained Hub mixture
python main.py --mode train \
  --model-name gpt2-large \
  --persona-mode pretrained-hub \
  --hf-base-repo user/bayesft \
  --mixture-mode pretrained-hub \
  --mixture-hub-repo user/bayesft-mixture \
  --num-personas 6 \
  --hf-token YOUR_TOKEN

# 4. Use pretrained personas + finetune mixture, then upload everything
python main.py --mode train \
  --model-name mistralai/Mistral-7B-v0.1 \
  --persona-mode pretrained-hub \
  --hf-base-repo user/bayesft \
  --mixture-mode finetune \
  --mixture-model-name mistralai/Mistral-7B-v0.1 \
  --save-to-hub \
  --hf-token YOUR_TOKEN \
  --num-personas 6

# 5. Inference using Hub models
python main.py --mode inference \
  --model-name gpt2-large \
  --hf-base-repo user/bayesft \
  --hf-token YOUR_TOKEN \
  --num-personas 6
"""
