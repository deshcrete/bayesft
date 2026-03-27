"""Log probability-based posterior inference pipeline."""

import argparse
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bayesft.core.dataset import DataSplitter, group_and_save_personas
from bayesft.core.posterior import LogProbPosterior
from bayesft.models.persona import PersonaLLM, PretrainedPersonaLLM


def setup_argparse():
    parser = argparse.ArgumentParser(description="BayesFT: Log probability-based inference")

    parser.add_argument("--mode", type=str, choices=["train", "inference"], required=True)
    parser.add_argument("--model-name", type=str, default="gpt2-large")

    # Persona stage
    parser.add_argument("--persona-mode", type=str,
                        choices=["finetune", "pretrained-hub"], default="finetune")
    parser.add_argument("--persona-hub-repos", type=str, nargs="+", default=None)

    # Mixture stage
    parser.add_argument("--mixture-mode", type=str,
                        choices=["finetune", "pretrained", "pretrained-hub"], default="pretrained")
    parser.add_argument("--mixture-model-name", type=str, default=None)
    parser.add_argument("--mixture-model-path", type=str, default="./uniform_prior")
    parser.add_argument("--mixture-hub-repo", type=str, default=None)
    parser.add_argument("--mixture-is-peft", action="store_true")

    # Shared
    parser.add_argument("--num-personas", type=int, default=6)
    parser.add_argument("--dataset-name", type=str, default="desh2806/bayesft-similar")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--hf-base-repo", type=str, default=None)
    parser.add_argument("--save-to-hub", action="store_true")

    # Posterior method
    parser.add_argument("--posterior-method", type=str, choices=["nnls", "slsqp"], default="nnls")

    # Skip options
    parser.add_argument("--posterior-only", action="store_true")
    parser.add_argument("--skip-personas", action="store_true")
    parser.add_argument("--logprob-dir", type=str, default="./data/logprobs/")

    return parser


def prepare_datasets(dataset_name="desh2806/bayesft-similar"):
    """Split dataset and group personas."""
    DataSplitter(
        dataset_name=dataset_name,
        data_split="train",
        split_names=["sft", "infer", "mixture", "personas"],
        split_sizes=[0.1, 0.1, 0.3, 0.5],
    ).split_data()
    group_and_save_personas("./data/sft")


def build_personas_finetune(args, inference_data):
    """Fine-tune a LoRA adapter for each persona."""
    personas = []
    for i in tqdm(range(args.num_personas), desc="Persona fine-tuning"):
        p = PersonaLLM(str(i), args.model_name)
        p.fine_tune()
        p.gen_logprobs(inference_data, f"./data/logprobs/persona_{i}")
        personas.append(p)
    return personas


def build_personas_pretrained_hub(args, inference_data):
    """Load pretrained PEFT persona adapters from HF Hub."""
    repos = args.persona_hub_repos
    if repos is None:
        if not args.hf_base_repo:
            raise ValueError("--persona-hub-repos or --hf-base-repo required")
        repos = [f"{args.hf_base_repo}-persona-{i}" for i in range(args.num_personas)]

    personas = []
    for i, repo in enumerate(tqdm(repos, desc="Loading persona adapters")):
        p = PretrainedPersonaLLM.from_peft_hub(args.model_name, repo, token=args.hf_token)
        p.gen_logprobs(inference_data, f"./data/logprobs/persona_{i}")
        personas.append(p)
    return personas


def build_mixture_pretrained(args, inference_data, logprob_dir="./data/logprobs/"):
    """Use a bare pretrained model as mixture."""
    mix_model = args.mixture_model_name or args.model_name
    mixture = PretrainedPersonaLLM.from_pretrained(args.mixture_model_path)
    mixture.gen_logprobs(inference_data, f"{logprob_dir}pretrain")
    return mixture


def build_mixture_finetune(args, inference_data, logprob_dir="./data/logprobs/"):
    """Fine-tune a LoRA mixture model."""
    mix_model = args.mixture_model_name or args.model_name
    mixture = PersonaLLM("pretrain", mix_model, is_mixture=True)
    mixture.fine_tune()
    mixture.gen_logprobs(inference_data, f"{logprob_dir}pretrain")
    return mixture


def build_mixture_pretrained_hub(args, inference_data, logprob_dir="./data/logprobs/"):
    """Load mixture from HF Hub."""
    if not args.mixture_hub_repo:
        raise ValueError("--mixture-hub-repo required for pretrained-hub mixture mode")
    mix_model = args.mixture_model_name or args.model_name

    if args.mixture_is_peft:
        mixture = PretrainedPersonaLLM.from_peft_hub(
            mix_model, args.mixture_hub_repo, token=args.hf_token
        )
    else:
        mixture = PretrainedPersonaLLM.from_pretrained(
            args.mixture_hub_repo, token=args.hf_token
        )

    mixture.gen_logprobs(inference_data, f"{logprob_dir}pretrain")
    return mixture


def solve_posterior(args):
    """Solve for posterior weights."""
    posterior = LogProbPosterior(args.num_personas, args.logprob_dir)
    posterior.construct_logprob_matrix()
    posterior.construct_logprob_vec()
    posterior.solve_for_weights(method=args.posterior_method)

    dist = posterior.weights
    print(f"Posterior weights: {dist}")
    posterior.save_weights("./posterior_weights.json")

    with open("./dists.txt", "a") as f:
        f.write("[" + ", ".join(str(x) for x in dist) + "]\n")

    return posterior, dist


def train_and_save_models(args):
    """Main training pipeline."""
    print("=== Starting Training Mode ===")

    if args.posterior_only:
        print(f"> Posterior-only mode: loading logprobs from {args.logprob_dir}")
        return solve_posterior(args)

    if not args.skip_personas:
        prepare_datasets(args.dataset_name)

    inference_data = load_from_disk("./data/infer")

    if not args.skip_personas:
        print(f"> Persona mode: {args.persona_mode}")
        if args.persona_mode == "finetune":
            personas = build_personas_finetune(args, inference_data)
        elif args.persona_mode == "pretrained-hub":
            personas = build_personas_pretrained_hub(args, inference_data)

    print(f"> Mixture mode: {args.mixture_mode}")
    if args.mixture_mode == "finetune":
        build_mixture_finetune(args, inference_data, args.logprob_dir)
    elif args.mixture_mode == "pretrained":
        build_mixture_pretrained(args, inference_data, args.logprob_dir)
    elif args.mixture_mode == "pretrained-hub":
        build_mixture_pretrained_hub(args, inference_data, args.logprob_dir)

    posterior, dist = solve_posterior(args)

    if args.save_to_hub and args.hf_base_repo:
        print("> Uploading to HuggingFace Hub")
        if not args.skip_personas:
            for i, persona in enumerate(personas):
                if isinstance(persona, PersonaLLM):
                    persona.save_to_hub(f"{args.hf_base_repo}-persona-{i}", token=args.hf_token)
        posterior.save_to_hub(args.hf_base_repo, token=args.hf_token)

    return dist


def load_and_infer(args):
    """Load models from Hub and display weights."""
    if not args.hf_base_repo:
        print("Error: --hf-base-repo required for inference mode")
        return

    posterior = LogProbPosterior(args.num_personas, args.logprob_dir)
    weights = posterior.load_from_hub(args.hf_base_repo, token=args.hf_token)
    print(f"Loaded posterior weights: {weights}")
    return weights


def main():
    parser = setup_argparse()
    args = parser.parse_args()

    if args.mode == "train":
        train_and_save_models(args)
    elif args.mode == "inference":
        load_and_infer(args)


if __name__ == "__main__":
    main()
