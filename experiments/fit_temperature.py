#!/usr/bin/env python3
"""
Fit the effective Bayesian temperature of SFT models.

For each mixed training condition, find T such that the tempered
mixture posterior predictive best matches the SFT model's logprobs:

    w_i(T) ∝ exp( (1/T) Σ_x log P_i(x) )
    P_pred(x; T) = Σ_i w_i(T) P_i(x)

    T* = argmin_T  MSE( SFT_logprobs, P_pred(·; T) )

T=1 is standard Bayes (mixture MAP), T→0 is hypothesis selection.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import gc

from bayesft.core.logprobs import gen_logprobs


PERSONAS = ["victorian", "counterculture", "techbro", "monk"]


def tempered_posterior(persona_logprobs_train, prior_weights, T):
    """Compute tempered Bayesian posterior on training data.

    w_i(T) ∝ prior_i · exp( (1/T) Σ_x log P_i(x) )
    """
    n = len(prior_weights)
    log_prior = np.log(prior_weights)
    log_lik = persona_logprobs_train.sum(axis=1)  # (n,)
    log_post = log_prior + log_lik / T
    return np.exp(log_post - logsumexp(log_post))


def bayesian_predictive(persona_logprobs_eval, weights):
    """log P_pred(x) = logsumexp(log w_i + log P_i(x))"""
    log_w = np.log(weights + 1e-30)[:, np.newaxis]
    return logsumexp(log_w + persona_logprobs_eval, axis=0)


def fit_temperature(persona_lps_train, persona_lps_eval, sft_lps, prior_weights):
    """Find T that minimizes MSE between tempered predictive and SFT logprobs."""

    def mse_at_T(log_T):
        T = np.exp(log_T)
        w = tempered_posterior(persona_lps_train, prior_weights, T)
        pred = bayesian_predictive(persona_lps_eval, w)
        return np.mean((sft_lps - pred) ** 2)

    # Search over log(T) from T=0.01 to T=100000
    result = minimize_scalar(mse_at_T, bounds=(-4, 12), method="bounded")
    best_T = np.exp(result.x)
    best_mse = result.fun
    best_w = tempered_posterior(persona_lps_train, prior_weights, best_T)

    return best_T, best_mse, best_w


def main():
    output_dir = "./hypothesis_test"
    model_name = "gpt2-large"
    n = len(PERSONAS)
    prior_weights = np.ones(n) / n

    # Define conditions
    conditions = [
        {"name": "victorian_pure", "adapter": "victorian",
         "train_data": "victorian_narrow", "true_weights": [1, 0, 0, 0]},
        {"name": "counterculture_pure", "adapter": "counterculture",
         "train_data": "counterculture_narrow", "true_weights": [0, 1, 0, 0]},
        {"name": "techbro_pure", "adapter": "techbro",
         "train_data": "techbro_narrow", "true_weights": [0, 0, 1, 0]},
        {"name": "monk_pure", "adapter": "monk",
         "train_data": "monk_narrow", "true_weights": [0, 0, 0, 1]},
        {"name": "70vic_30monk", "adapter": "mixed_70vic_30monk",
         "train_data": None, "true_weights": [0.7, 0, 0, 0.3]},
        {"name": "50vic_50counter", "adapter": "mixed_50vic_50counter",
         "train_data": None, "true_weights": [0.5, 0.5, 0, 0]},
        {"name": "50tech_50monk", "adapter": "mixed_50tech_50monk",
         "train_data": None, "true_weights": [0, 0, 0.5, 0.5]},
    ]

    # Load base model
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Load broad eval data
    all_broad = []
    for name in PERSONAS:
        ds = load_from_disk(f"{output_dir}/data/{name}_broad")
        all_broad.append(ds)
    from datasets import concatenate_datasets
    broad_eval = concatenate_datasets(all_broad)
    print(f"Broad eval: {len(broad_eval)} examples")

    # Compute persona logprobs on broad eval (reuse across conditions)
    print("Computing persona logprobs on broad eval...")
    persona_eval_lps = []
    for name in PERSONAS:
        adapter_path = f"{output_dir}/adapters/{name}"
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        ds = gen_logprobs(model, tokenizer, broad_eval)
        persona_eval_lps.append(np.array(ds["completion_logprob"]))
        model.unload()
        del model
        torch.cuda.empty_cache()
    persona_eval_lps = np.array(persona_eval_lps)

    # Process each condition
    results = []
    for cond in conditions:
        print(f"\n{'='*50}")
        print(f"Condition: {cond['name']}")
        print(f"{'='*50}")

        true_w = np.array(cond["true_weights"])

        # Load training data
        if cond["train_data"]:
            train_ds = load_from_disk(f"{output_dir}/data/{cond['train_data']}")
        else:
            # Mixed conditions don't have separate saved train data
            # Reconstruct from component narrow data
            from datasets import Dataset
            rng = np.random.default_rng(42)
            total = 150
            rows = []
            for i, name in enumerate(PERSONAS):
                if true_w[i] > 0:
                    narrow = load_from_disk(f"{output_dir}/data/{name}_narrow")
                    count = int(true_w[i] * total)
                    indices = rng.choice(len(narrow), size=count, replace=True)
                    for idx in indices:
                        rows.append(narrow[int(idx)])
            train_ds = Dataset.from_list(rows)

        # Compute persona logprobs on training data
        print(f"  Computing persona logprobs on training data ({len(train_ds)} examples)...")
        persona_train_lps = []
        for name in PERSONAS:
            adapter_path = f"{output_dir}/adapters/{name}"
            model = PeftModel.from_pretrained(base_model, adapter_path)
            model.eval()
            ds = gen_logprobs(model, tokenizer, train_ds)
            persona_train_lps.append(np.array(ds["completion_logprob"]))
            model.unload()
            del model
            torch.cuda.empty_cache()
        persona_train_lps = np.array(persona_train_lps)

        # Compute SFT model logprobs on broad eval
        print(f"  Computing SFT logprobs on broad eval...")
        adapter_path = f"{output_dir}/adapters/{cond['adapter']}"
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        ds = gen_logprobs(model, tokenizer, broad_eval)
        sft_lps = np.array(ds["completion_logprob"])
        model.unload()
        del model
        torch.cuda.empty_cache()

        # Fit temperature
        best_T, best_mse, best_w = fit_temperature(
            persona_train_lps, persona_eval_lps, sft_lps, prior_weights
        )

        # Compare against fixed baselines
        w_t1 = tempered_posterior(persona_train_lps, prior_weights, 1.0)
        pred_t1 = bayesian_predictive(persona_eval_lps, w_t1)
        mse_t1 = np.mean((sft_lps - pred_t1) ** 2)

        pred_true = bayesian_predictive(persona_eval_lps, true_w)
        mse_true = np.mean((sft_lps - pred_true) ** 2)

        # Data entropy (bits)
        nonzero = true_w[true_w > 0]
        data_entropy = -np.sum(nonzero * np.log2(nonzero))

        print(f"  True weights:     {true_w}")
        print(f"  Best T:           {best_T:.2f}")
        print(f"  Best weights:     {best_w}")
        print(f"  Best MSE:         {best_mse:.1f}")
        print(f"  MSE at T=1:       {mse_t1:.1f}")
        print(f"  MSE true weights: {mse_true:.1f}")
        print(f"  Data entropy:     {data_entropy:.3f} bits")

        # Temperature sweep for this condition
        temps = np.logspace(-2, 5, 200)
        sweep_mses = []
        for T in temps:
            w = tempered_posterior(persona_train_lps, prior_weights, T)
            pred = bayesian_predictive(persona_eval_lps, w)
            sweep_mses.append(np.mean((sft_lps - pred) ** 2))

        results.append({
            "name": cond["name"],
            "true_weights": true_w.tolist(),
            "best_T": float(best_T),
            "best_weights": best_w.tolist(),
            "best_mse": float(best_mse),
            "mse_t1": float(mse_t1),
            "mse_true": float(mse_true),
            "data_entropy": float(data_entropy),
            "sweep_temps": temps.tolist(),
            "sweep_mses": sweep_mses,
        })

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Save results ──
    with open(f"{output_dir}/temperature_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Plots ──
    plot_dir = f"{output_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Plot 1: Temperature sweep curves
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in results:
        ax.plot(r["sweep_temps"], r["sweep_mses"], label=r["name"], linewidth=1.5)
        ax.axvline(r["best_T"], color="gray", linestyle=":", alpha=0.3)
        ax.scatter([r["best_T"]], [r["best_mse"]], s=60, zorder=5, edgecolors="black")
    ax.set_xscale("log")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("MSE (SFT logprobs vs tempered predictive)")
    ax.set_title("MSE vs Bayesian Temperature\nEach curve is a training condition")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(0.01, 1e5)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/temperature_sweep.png", dpi=150)
    plt.close()
    print("Saved temperature_sweep.png")

    # Plot 2: Best T vs data entropy
    fig, ax = plt.subplots(figsize=(8, 6))
    entropies = [r["data_entropy"] for r in results]
    best_Ts = [r["best_T"] for r in results]
    names = [r["name"] for r in results]

    ax.scatter(entropies, best_Ts, s=100, c="steelblue", edgecolors="black", zorder=3)
    for i, name in enumerate(names):
        ax.annotate(name, (entropies[i], best_Ts[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)

    ax.set_xlabel("Training data entropy (bits)")
    ax.set_ylabel("Best-fit temperature T")
    ax.set_yscale("log")
    ax.set_title("Effective Bayesian Temperature vs Data Entropy\n"
                 "Higher entropy (more balanced data) → higher T (softer posterior)")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/temperature_vs_entropy.png", dpi=150)
    plt.close()
    print("Saved temperature_vs_entropy.png")

    # Plot 3: Summary table
    print(f"\n{'='*70}")
    print(f"{'Condition':<22s}  {'Entropy':>7s}  {'Best T':>8s}  {'MSE(T*)':>8s}  {'MSE(T=1)':>8s}  {'Weights at T*'}")
    print(f"{'='*70}")
    for r in results:
        w_str = " ".join(f"{x:.3f}" for x in r["best_weights"])
        print(f"{r['name']:<22s}  {r['data_entropy']:7.3f}  {r['best_T']:8.2f}  "
              f"{r['best_mse']:8.1f}  {r['mse_t1']:8.1f}  [{w_str}]")

    print(f"\nAll results saved to {output_dir}/temperature_results.json")
    print(f"Plots saved to {plot_dir}/")


if __name__ == "__main__":
    main()
