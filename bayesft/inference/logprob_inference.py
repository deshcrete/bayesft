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
    parser.add_argument("--posterior-method", type=str,
                        choices=["nnls", "slsqp", "standardized", "standardized-reg",
                                 "delta", "delta-reg"],
                        default="nnls")
    parser.add_argument("--reg-lambda", type=float, default=0.1,
                        help="Regularization strength for standardized-reg method")

    # Skip options
    parser.add_argument("--posterior-only", action="store_true")
    parser.add_argument("--skip-personas", action="store_true",
                        help="Skip persona training AND logprob generation (use cached)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip persona training but regenerate logprobs from existing adapters")
    parser.add_argument("--logprob-dir", type=str, default="./data/logprobs/")
    parser.add_argument("--adapter-dir", type=str, default="./lora_weights",
                        help="Base directory for saving/loading LoRA adapters")
    parser.add_argument("--infer-samples", type=int, default=None,
                        help="Subsample inference data to this many examples (default: use all)")

    # LoRA hyperparameters
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--num-epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")

    # Split ratios
    parser.add_argument("--persona-ratio", type=float, default=0.5,
                        help="Fraction of data for persona training (default 0.5 = ~5k per persona)")
    parser.add_argument("--infer-ratio", type=float, default=0.1,
                        help="Fraction of data for inference")
    parser.add_argument("--mixture-ratio", type=float, default=0.3,
                        help="Fraction of data for mixture training")

    return parser


def prepare_datasets(dataset_name="desh2806/bayesft-similar", persona_ratio=0.5, infer_ratio=0.1, mixture_ratio=0.3):
    """Split dataset and group personas.

    Splits:
        - personas: used for training persona LoRAs (split by persona)
        - infer: used for logprob evaluation
        - mixture: used for training the mixture model
        - unused: remainder
    """
    unused_ratio = 1.0 - persona_ratio - infer_ratio - mixture_ratio

    print(f"> Split ratios: personas={persona_ratio}, infer={infer_ratio}, "
          f"mixture={mixture_ratio}, unused={unused_ratio:.2f}")

    DataSplitter(
        dataset_name=dataset_name,
        data_split="train",
        split_names=["personas", "infer", "mixture", "unused"],
        split_sizes=[persona_ratio, infer_ratio, mixture_ratio, unused_ratio],
    ).split_data()
    group_and_save_personas("./data/personas")


def build_personas_finetune(args, inference_data):
    """Fine-tune a LoRA adapter for each persona, cleaning up between each."""
    import gc
    personas = []
    for i in tqdm(range(args.num_personas), desc="Persona fine-tuning"):
        p = PersonaLLM(str(i), args.model_name, adapter_dir=args.adapter_dir)

        model, tokenizer = p.fine_tune(
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            epochs=args.num_epochs, lr=args.lr,
        )
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        p.gen_logprobs(inference_data, f"{args.logprob_dir}persona_{i}")
        gc.collect()
        torch.cuda.empty_cache()

        personas.append(p)
    return personas


def gen_base_logprobs(args, inference_data):
    """Compute logprobs from the base model (no LoRA) on inference data."""
    import gc
    from bayesft.core.logprobs import gen_logprobs as _gen_logprobs

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    base_model.eval()

    print("> Computing base model logprobs...")
    _gen_logprobs(base_model, tokenizer, inference_data,
                  output_path=f"{args.logprob_dir}base")

    del base_model
    gc.collect()
    torch.cuda.empty_cache()


def regen_persona_logprobs(args, inference_data):
    """Regenerate logprobs from existing trained adapters (no training).

    Loads base model once, loads each adapter one at a time, computes logprobs, unloads.
    Also computes base model logprobs for delta-based methods.
    """
    import gc
    from peft import PeftModel as _PeftModel
    from bayesft.core.logprobs import gen_logprobs as _gen_logprobs

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Base model logprobs (needed for delta methods)
    print("> Computing base model logprobs...")
    base_model.eval()
    _gen_logprobs(base_model, tokenizer, inference_data,
                  output_path=f"{args.logprob_dir}base")

    for i in tqdm(range(args.num_personas), desc="Persona logprobs"):
        adapter_path = f"{args.adapter_dir}/models/persona_{i}"
        model = _PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        _gen_logprobs(model, tokenizer, inference_data,
                      output_path=f"{args.logprob_dir}persona_{i}")
        model.unload()
        del model
        torch.cuda.empty_cache()

    del base_model
    gc.collect()
    torch.cuda.empty_cache()


def build_personas_pretrained_hub(args, inference_data):
    """Load pretrained PEFT persona adapters from HF Hub."""
    import gc
    repos = args.persona_hub_repos
    if repos is None:
        if not args.hf_base_repo:
            raise ValueError("--persona-hub-repos or --hf-base-repo required")
        repos = [f"{args.hf_base_repo}-persona-{i}" for i in range(args.num_personas)]

    personas = []
    for i, repo in enumerate(tqdm(repos, desc="Loading persona adapters")):
        p = PretrainedPersonaLLM.from_peft_hub(args.model_name, repo, token=args.hf_token)
        p.gen_logprobs(inference_data, f"{args.logprob_dir}persona_{i}")
        # Free model after logprobs are saved
        del p.model
        gc.collect()
        torch.cuda.empty_cache()
        personas.append(p)
    return personas


def build_mixture_pretrained(args, inference_data, logprob_dir="./data/logprobs/"):
    """Use a bare pretrained model as mixture."""
    import gc
    mix_model = args.mixture_model_name or args.model_name
    mixture = PretrainedPersonaLLM.from_local(args.mixture_model_path, tokenizer_name=mix_model)
    mixture.gen_logprobs(inference_data, f"{logprob_dir}pretrain")
    del mixture
    gc.collect()
    torch.cuda.empty_cache()


def build_mixture_finetune(args, inference_data, logprob_dir="./data/logprobs/"):
    """Fine-tune a LoRA mixture model and save merged model."""
    import gc
    mix_model = args.mixture_model_name or args.model_name
    mixture = PersonaLLM("pretrain", mix_model, is_mixture=True, adapter_dir=args.adapter_dir)
    model, tokenizer = mixture.fine_tune(
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        epochs=args.num_epochs, lr=args.lr,
    )

    # Save merged model (base + LoRA) for future use as pretrained mixture
    if args.mixture_model_path:
        print(f"> Merging and saving mixture model to {args.mixture_model_path}")
        merged = model.merge_and_unload()
        merged.save_pretrained(args.mixture_model_path)
        tokenizer.save_pretrained(args.mixture_model_path)
        print(f"  Saved merged mixture model to {args.mixture_model_path}")
        del merged

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    mixture.gen_logprobs(inference_data, f"{logprob_dir}pretrain")
    gc.collect()
    torch.cuda.empty_cache()
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

    if args.posterior_method in ("delta", "delta-reg"):
        posterior.construct_base_logprobs()

    posterior.solve_for_weights(method=args.posterior_method, lam=args.reg_lambda)

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

    if not args.skip_personas and not args.skip_training:
        prepare_datasets(args.dataset_name, persona_ratio=args.persona_ratio,
                         infer_ratio=args.infer_ratio, mixture_ratio=args.mixture_ratio)

    inference_data = load_from_disk("./data/infer")
    if args.infer_samples and args.infer_samples < len(inference_data):
        inference_data = inference_data.shuffle().select(range(args.infer_samples))
        print(f"> Subsampled inference data to {len(inference_data)} examples")

    if args.skip_training:
        # Regenerate logprobs from existing adapters (no training)
        print("> Regenerating persona logprobs from existing adapters...")
        regen_persona_logprobs(args, inference_data)
    elif not args.skip_personas:
        print(f"> Persona mode: {args.persona_mode}")
        if args.persona_mode == "finetune":
            personas = build_personas_finetune(args, inference_data)
        elif args.persona_mode == "pretrained-hub":
            personas = build_personas_pretrained_hub(args, inference_data)

    # Compute base model logprobs for delta methods (if not already done by regen)
    if args.posterior_method in ("delta", "delta-reg") and not args.skip_training:
        import os
        base_path = f"{args.logprob_dir}base"
        if not os.path.exists(base_path):
            gen_base_logprobs(args, inference_data)

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
