#!/usr/bin/env python3
"""
Test whether SFT acts as approximate Bayesian updating.

Protocol:
  1. Compute analytic Bayesian posterior weights from persona logprobs
     on skewed training data (e.g. 50% persona 0, 10% each rest).
  2. Fine-tune the prior model (uniform_prior) on the same skewed data.
  3. Compute logprobs from the fine-tuned model on held-out data.
  4. Compare against:
     a) Bayesian posterior predictive: Σ w_posterior_i P_i(x)
     b) Prior predictive:             Σ w_prior_i P_i(x)   (uniform)
  5. If SFT ≈ Bayesian, the fine-tuned model's logprobs should be
     closer to (a) than to (b).

Usage:
    # Step 1: Fine-tune shifted model (only needed once)
    python experiments/bayesian_test.py --step finetune \
        --target-persona 0 --target-weight 0.5

    # Step 2: Evaluate (can re-run with different metrics)
    python experiments/bayesian_test.py --step evaluate \
        --target-persona 0 --target-weight 0.5
"""

import argparse
import gc
import json
import os
from collections import Counter
from functools import partial

import numpy as np
import torch
from scipy.special import logsumexp
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bayesft.core.logprobs import gen_logprobs
from bayesft.core.sft import fine_tune, tokenize_prompt_completion


def make_skewed_dataset(dataset, persona_names, target_idx, target_weight, total_samples=3000, seed=42):
    """Resample dataset to have skewed persona distribution.

    Args:
        dataset: Full dataset with 'persona' column.
        persona_names: List of unique persona names (ordered).
        target_idx: Index of persona to upweight.
        target_weight: Weight for target persona (rest split equally).
        total_samples: Total examples in resampled dataset.
        seed: Random seed.

    Returns:
        Resampled dataset, true weight vector.
    """
    n = len(persona_names)
    other_weight = (1.0 - target_weight) / (n - 1)
    true_weights = np.array([other_weight] * n)
    true_weights[target_idx] = target_weight

    rng = np.random.default_rng(seed)

    # Group indices by persona
    persona_to_indices = {p: [] for p in persona_names}
    for i, ex in enumerate(dataset):
        persona_to_indices[ex["persona"]].append(i)

    # Sample according to weights
    all_indices = []
    for i, p in enumerate(persona_names):
        count = int(true_weights[i] * total_samples)
        available = persona_to_indices[p]
        chosen = rng.choice(available, size=count, replace=len(available) < count)
        all_indices.extend(chosen.tolist())

    rng.shuffle(all_indices)
    return dataset.select(all_indices), true_weights


def compute_single_posterior(persona_logprobs, prior_weights, temperature=1.0):
    """Compute posterior over 'which single persona generated all data'.

    P(persona_i | data) ∝ prior_i · Π_x P_i(x)^(1/T)

    This is the WRONG model for mixtures — it asks 'which one persona?'
    rather than 'what mixture weights?'. Kept for comparison.
    """
    log_prior = np.log(prior_weights)
    log_likelihood = persona_logprobs.sum(axis=1)
    log_posterior = log_prior + log_likelihood / temperature
    posterior = np.exp(log_posterior - logsumexp(log_posterior))
    return posterior


def compute_mixture_posterior(persona_logprobs, prior_weights, n_steps=500, lr=0.05):
    """Compute MAP mixture weights via gradient descent on mixture log-likelihood.

    Maximizes:
        L(w) = Σ_x log( Σ_i w_i P_i(x) )
             = Σ_x logsumexp_i( log w_i + log P_i(x) )

    subject to w_i ≥ 0, Σ w_i = 1 (enforced via softmax parameterization).

    This is the CORRECT mixture model — each example can come from any
    persona, and we infer the mixing proportions.

    Args:
        persona_logprobs: (n_personas, n_samples) matrix of log P_i(x).
        prior_weights: (n_personas,) prior (used as initialization).
        n_steps: Gradient descent steps.
        lr: Learning rate.

    Returns:
        posterior_weights: (n_personas,) MAP mixture weights.
    """
    import torch

    n, m = persona_logprobs.shape
    lps = torch.tensor(persona_logprobs, dtype=torch.float64)  # (n, m)

    # Initialize logits from prior
    logits = torch.tensor(np.log(prior_weights + 1e-30), dtype=torch.float64, requires_grad=True)

    optimizer = torch.optim.Adam([logits], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()
        log_w = torch.log_softmax(logits, dim=0)  # (n,)
        # log P(x | w) = logsumexp_i(log w_i + log P_i(x))
        log_mix = torch.logsumexp(log_w[:, None] + lps, dim=0)  # (m,)
        # Negative log-likelihood
        nll = -log_mix.sum()
        nll.backward()
        optimizer.step()

        if step % 100 == 0 or step == n_steps - 1:
            w = torch.softmax(logits, dim=0).detach().numpy()
            print(f"    Step {step:4d}: NLL={nll.item():.1f}  w={w}")

    weights = torch.softmax(logits, dim=0).detach().numpy()
    return weights


def bayesian_predictive(persona_logprobs, weights):
    """Compute Bayesian predictive logprobs: log Σ_i w_i P_i(x).

    Args:
        persona_logprobs: (n_personas, n_samples) matrix.
        weights: (n_personas,) weight vector.

    Returns:
        (n_samples,) predictive logprobs.
    """
    # log(Σ w_i P_i(x)) = logsumexp(log w_i + log P_i(x))
    log_weights = np.log(weights + 1e-30)[:, np.newaxis]  # (n, 1)
    return logsumexp(log_weights + persona_logprobs, axis=0)  # (m,)


def step_finetune(args):
    """Fine-tune the prior model on skewed data."""
    print("=" * 60)
    print("STEP 1: Fine-tune on skewed data")
    print("=" * 60)

    # Load full dataset
    full_dataset = load_dataset(args.source_dataset, split="train")
    persona_names = sorted(set(full_dataset["persona"]))
    n = len(persona_names)
    print(f"Found {n} personas, {len(full_dataset)} total examples")

    # Create skewed dataset
    skewed, true_weights = make_skewed_dataset(
        full_dataset, persona_names, args.target_persona,
        args.target_weight, args.finetune_samples
    )
    print(f"\nSkewed dataset: {len(skewed)} examples")
    print(f"True weights: {true_weights}")
    counts = Counter(skewed["persona"])
    for i, p in enumerate(persona_names):
        print(f"  P{i}: {counts.get(p, 0)} examples ({counts.get(p, 0)/len(skewed):.3f})")

    # Save skewed dataset for reference
    os.makedirs(args.output_dir, exist_ok=True)
    skewed.save_to_disk(f"{args.output_dir}/skewed_data")
    with open(f"{args.output_dir}/config.json", "w") as f:
        json.dump({
            "true_weights": true_weights.tolist(),
            "target_persona": args.target_persona,
            "target_weight": args.target_weight,
            "finetune_samples": args.finetune_samples,
        }, f, indent=2)

    # Fine-tune
    print(f"\nFine-tuning {args.base_model} on skewed data...")
    shifted_path = f"{args.output_dir}/shifted_model"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model, _ = fine_tune(
        model_name=args.base_model,
        dataset=skewed,
        output_path=shifted_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        epochs=args.finetune_epochs,
        lr=args.finetune_lr,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Shifted model saved to {shifted_path}")


def step_evaluate(args):
    """Evaluate: compare SFT model vs Bayesian posterior predictive."""
    print("=" * 60)
    print("STEP 2: Evaluate SFT vs Bayesian posterior")
    print("=" * 60)

    n = args.num_personas

    # Load config
    with open(f"{args.output_dir}/config.json") as f:
        config = json.load(f)
    true_weights = np.array(config["true_weights"])
    print(f"True weights: {true_weights}")

    # Load persona logprobs on eval data
    print(f"\nLoading persona logprobs from {args.logprob_dir}...")
    persona_lps = []
    for i in range(n):
        ds = load_from_disk(f"{args.logprob_dir}persona_{i}")
        persona_lps.append(np.array(ds["completion_logprob"]))
    persona_lps = np.array(persona_lps)  # (n, m)
    m = persona_lps.shape[1]
    print(f"  {n} personas, {m} eval examples")

    prior_weights = np.ones(n) / n

    # Single-persona posterior (wrong model, for reference)
    single_posterior = compute_single_posterior(persona_lps, prior_weights)
    print(f"\nSingle-persona posterior:   {single_posterior}")
    print(f"True weights:              {true_weights}")

    # Mixture posterior on TRAINING data
    # We need persona logprobs on the skewed data the SFT model was trained on.
    print(f"\nComputing persona logprobs on skewed training data...")
    skewed_data = load_from_disk(f"{args.output_dir}/skewed_data")
    print(f"  Skewed data: {len(skewed_data)} examples")

    from peft import PeftModel as _PeftModel
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    train_persona_lps = []
    for i in range(n):
        adapter_path = f"./lora_weights_gpt2/models/persona_{i}"
        model = _PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        ds = gen_logprobs(model, tokenizer, skewed_data)
        train_persona_lps.append(np.array(ds["completion_logprob"]))
        model.unload()
        del model
        torch.cuda.empty_cache()

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    train_persona_lps = np.array(train_persona_lps)  # (n, n_train)
    print(f"  Train persona logprobs: {train_persona_lps.shape}")

    print(f"\nComputing mixture MAP posterior on training data...")
    mixture_posterior = compute_mixture_posterior(train_persona_lps, prior_weights)
    print(f"Mixture MAP posterior:     {mixture_posterior}")
    print(f"True weights:              {true_weights}")

    # Compute predictive distributions
    prior_predictive = bayesian_predictive(persona_lps, prior_weights)
    true_predictive = bayesian_predictive(persona_lps, true_weights)
    single_predictive = bayesian_predictive(persona_lps, single_posterior)
    mixture_predictive = bayesian_predictive(persona_lps, mixture_posterior)

    # Load SFT model and compute its logprobs on eval data
    print(f"\nComputing SFT model logprobs...")
    shifted_path = f"{args.output_dir}/shifted_model"
    eval_data = load_from_disk("./data/infer")
    if args.eval_samples and args.eval_samples < len(eval_data):
        eval_data = eval_data.shuffle(seed=42).select(range(args.eval_samples))

    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, shifted_path)
    model.eval()

    sft_ds = gen_logprobs(model, tokenizer, eval_data)
    sft_lps = np.array(sft_ds["completion_logprob"])

    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()

    # Truncate to same length
    min_len = min(m, len(sft_lps))
    persona_lps = persona_lps[:, :min_len]
    prior_predictive = prior_predictive[:min_len]
    true_predictive = true_predictive[:min_len]
    single_predictive = single_predictive[:min_len]
    mixture_predictive = mixture_predictive[:min_len]
    sft_lps = sft_lps[:min_len]

    # --- Comparison metrics ---
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    comparisons = [
        ("Prior (uniform)", prior_predictive, prior_weights),
        ("True weights", true_predictive, true_weights),
        ("Single-persona posterior", single_predictive, single_posterior),
        ("Mixture MAP posterior", mixture_predictive, mixture_posterior),
    ]

    print(f"\n{'Method':<30s}  {'MSE':>10s}  {'Corr':>8s}  {'Mean LP':>8s}  Weights")
    for name, pred, weights in comparisons:
        mse = np.mean((sft_lps - pred) ** 2)
        corr = np.corrcoef(sft_lps, pred)[0, 1]
        w_str = " ".join(f"{x:.3f}" for x in weights)
        print(f"{name:<30s}  {mse:10.1f}  {corr:.6f}  {pred.mean():8.1f}  [{w_str}]")

    print(f"\n{'SFT model':<30s}  {'':>10s}  {'':>8s}  {sft_lps.mean():8.1f}")

    # Per-persona correlations
    print(f"\nSFT model correlation with each persona:")
    for i in range(n):
        r = np.corrcoef(sft_lps, persona_lps[i])[0, 1]
        print(f"  P{i}: {r:.4f}  (true w={true_weights[i]:.2f}, mixture w={mixture_posterior[i]:.3f})")

    # Save results
    results = {
        "true_weights": true_weights.tolist(),
        "single_posterior": single_posterior.tolist(),
        "mixture_posterior": mixture_posterior.tolist(),
        "comparisons": {
            name: {"mse": float(np.mean((sft_lps - pred) ** 2)),
                   "corr": float(np.corrcoef(sft_lps, pred)[0, 1]),
                   "weights": w.tolist()}
            for name, pred, w in comparisons
        },
    }
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["finetune", "evaluate", "both"], default="both")
    parser.add_argument("--model-name", type=str, default="gpt2-large")
    parser.add_argument("--base-model", type=str, default="gpt2-large",
                        help="Prior model to fine-tune (base model if no prior LoRA)")
    parser.add_argument("--num-personas", type=int, default=6)
    parser.add_argument("--target-persona", type=int, default=0)
    parser.add_argument("--target-weight", type=float, default=0.5)
    parser.add_argument("--source-dataset", type=str, default="desh2806/bayesft-similar")
    parser.add_argument("--logprob-dir", type=str, default="./data/logprobs/")
    parser.add_argument("--output-dir", type=str, default="./bayesian_test")
    parser.add_argument("--finetune-samples", type=int, default=3000)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--finetune-lr", type=float, default=5e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Fixed Bayesian temperature (default: sweep to find best)")
    args = parser.parse_args()

    if args.step in ("finetune", "both"):
        step_finetune(args)
    if args.step in ("evaluate", "both"):
        step_evaluate(args)


if __name__ == "__main__":
    main()
