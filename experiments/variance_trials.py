#!/usr/bin/env python3
"""
Variance analysis: resample inference sets and re-solve posterior.

Loads base model and mixture model ONCE, then for each trial:
  1. Samples N random examples from the full HF dataset
  2. Loads each persona adapter one at a time, computes logprobs, unloads
  3. Computes mixture logprobs
  4. Solves posterior weights
  5. Records the weights

Usage:
    python experiments/variance_trials.py \
        --model-name google/gemma-3-270m \
        --mixture-model-path ./uniform_prior \
        --num-personas 6 \
        --num-trials 20 \
        --infer-samples 300 \
        --results-file ./variance_results.json
"""

import argparse
import gc
import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from bayesft.core.logprobs import gen_logprobs
from bayesft.core.posterior import LogProbPosterior


def parse_args():
    parser = argparse.ArgumentParser(description="Variance analysis over resampled inference sets")
    parser.add_argument("--model-name", type=str, default="google/gemma-3-270m")
    parser.add_argument("--mixture-model-path", type=str, default="./uniform_prior")
    parser.add_argument("--mixture-tokenizer", type=str, default=None,
                        help="Tokenizer for mixture model (defaults to --model-name)")
    parser.add_argument("--num-personas", type=int, default=6)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--infer-samples", type=int, default=300)
    parser.add_argument("--dataset-name", type=str, default="desh2806/bayesft-similar")
    parser.add_argument("--adapter-dir", type=str, default="./lora_weights/models")
    parser.add_argument("--logprob-dir", type=str, default="./data/logprobs/")
    parser.add_argument("--posterior-method", type=str,
                        choices=["nnls", "slsqp", "standardized", "standardized-reg",
                                 "delta", "delta-reg"],
                        default="nnls")
    parser.add_argument("--reg-lambda", type=float, default=0.1,
                        help="Regularization strength for *-reg methods")
    parser.add_argument("--results-file", type=str, default="./variance_results.json")
    return parser.parse_args()


def run_trial(trial_num, inference_data, tokenizer, base_model, mix_tokenizer, mix_model, args):
    """Run one trial: load each adapter, compute logprobs, unload, then solve."""

    # Base model logprobs (needed for delta methods)
    if args.posterior_method in ("delta", "delta-reg"):
        base_model.eval()
        gen_logprobs(base_model, tokenizer, inference_data,
                     output_path=f"{args.logprob_dir}base")

    # Persona logprobs — load/unload each adapter one at a time
    for i in range(args.num_personas):
        adapter_path = f"{args.adapter_dir}/persona_{i}"
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        gen_logprobs(model, tokenizer, inference_data,
                     output_path=f"{args.logprob_dir}persona_{i}")
        # Delete PEFT wrapper to restore clean base model
        model.unload()
        del model
        torch.cuda.empty_cache()

    # Mixture logprobs
    gen_logprobs(mix_model, mix_tokenizer, inference_data,
                 output_path=f"{args.logprob_dir}pretrain")

    # Solve posterior
    posterior = LogProbPosterior(args.num_personas, args.logprob_dir)
    posterior.construct_logprob_matrix()
    posterior.construct_logprob_vec()
    if args.posterior_method in ("delta", "delta-reg"):
        posterior.construct_base_logprobs()
    posterior.solve_for_weights(method=args.posterior_method, lam=args.reg_lambda)

    return posterior.weights


def main():
    args = parse_args()

    print("=" * 60)
    print("VARIANCE ANALYSIS")
    print("=" * 60)
    print(f"Trials: {args.num_trials}")
    print(f"Samples per trial: {args.infer_samples}")
    print(f"Posterior method: {args.posterior_method}")
    print(f"Model: {args.model_name}")
    print(f"Mixture: {args.mixture_model_path}")

    # Load dataset once
    print(f"\n> Loading dataset: {args.dataset_name}")
    full_dataset = load_dataset(args.dataset_name, split="train")
    print(f"  Full dataset size: {len(full_dataset)}")

    # Load base model once (adapters will be loaded/unloaded per persona)
    print("> Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Load mixture model once
    print("> Loading mixture model...")
    mix_tokenizer_name = args.mixture_tokenizer or args.model_name
    mix_tokenizer = AutoTokenizer.from_pretrained(mix_tokenizer_name)
    mix_tokenizer.pad_token = mix_tokenizer.eos_token
    mix_model = AutoModelForCausalLM.from_pretrained(
        args.mixture_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    mix_model.eval()

    # Run trials
    all_weights = []
    for trial in range(args.num_trials):
        print(f"\n{'='*50}")
        print(f"TRIAL {trial + 1}/{args.num_trials}")
        print(f"{'='*50}")

        # Random subsample
        indices = random.sample(range(len(full_dataset)), args.infer_samples)
        inference_data = full_dataset.select(indices)
        print(f"  Sampled {len(inference_data)} examples")

        weights = run_trial(trial, inference_data, tokenizer, base_model,
                            mix_tokenizer, mix_model, args)
        all_weights.append(weights.tolist())
        print(f"  Weights: {weights}")

    # Cleanup
    del base_model, mix_model
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    all_weights_np = np.array(all_weights)
    mean_w = all_weights_np.mean(axis=0)
    std_w = all_weights_np.std(axis=0)
    expected = 1.0 / args.num_personas

    print(f"\n{'='*60}")
    print(f"SUMMARY ({args.num_trials} trials, {args.infer_samples} samples each)")
    print(f"{'='*60}")
    print(f"Expected (uniform): {expected:.4f}")
    print(f"Mean weights:       {mean_w}")
    print(f"Std weights:        {std_w}")
    print(f"Max |mean - uniform|: {np.max(np.abs(mean_w - expected)):.4f}")
    print(f"Mean std:           {std_w.mean():.4f}")

    for i in range(args.num_personas):
        print(f"  Persona {i}: {mean_w[i]:.4f} ± {std_w[i]:.4f}")

    # Save
    results = {
        "num_trials": args.num_trials,
        "infer_samples": args.infer_samples,
        "num_personas": args.num_personas,
        "posterior_method": args.posterior_method,
        "model_name": args.model_name,
        "mixture_model_path": args.mixture_model_path,
        "trials": all_weights,
        "mean": mean_w.tolist(),
        "std": std_w.tolist(),
    }

    if os.path.exists(args.results_file):
        with open(args.results_file, "r") as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            existing = [existing]
    else:
        existing = []

    existing.append(results)
    with open(args.results_file, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\nResults saved to {args.results_file}")


if __name__ == "__main__":
    main()
