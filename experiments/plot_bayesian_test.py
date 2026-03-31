#!/usr/bin/env python3
"""Generate plots for Bayesian test results."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from bayesft.core.logprobs import gen_logprobs


def load_all(logprob_dir, num_personas, output_dir, model_name, base_model_path, eval_samples=None):
    """Load all logprobs needed for plotting."""
    # Persona logprobs
    persona_lps = []
    for i in range(num_personas):
        ds = load_from_disk(f"{logprob_dir}persona_{i}")
        persona_lps.append(np.array(ds["completion_logprob"]))
    persona_lps = np.array(persona_lps)

    # SFT model logprobs
    shifted_path = f"{output_dir}/shifted_model"
    eval_data = load_from_disk("./data/infer")
    if eval_samples and eval_samples < len(eval_data):
        eval_data = eval_data.shuffle(seed=42).select(range(eval_samples))

    # Check if we have cached SFT logprobs
    sft_cache = f"{output_dir}/sft_logprobs"
    try:
        sft_ds = load_from_disk(sft_cache)
        sft_lps = np.array(sft_ds["completion_logprob"])
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, shifted_path)
        model.eval()
        sft_ds = gen_logprobs(model, tokenizer, eval_data, output_path=sft_cache)
        sft_lps = np.array(sft_ds["completion_logprob"])
        del model, base_model
        torch.cuda.empty_cache()

    # Truncate to match
    min_len = min(persona_lps.shape[1], len(sft_lps))
    persona_lps = persona_lps[:, :min_len]
    sft_lps = sft_lps[:min_len]

    return persona_lps, sft_lps


def main():
    output_dir = "./bayesian_test"
    logprob_dir = "./data/logprobs/"
    num_personas = 6
    model_name = "gpt2-large"

    with open(f"{output_dir}/config.json") as f:
        config = json.load(f)
    with open(f"{output_dir}/results.json") as f:
        results = json.load(f)

    true_weights = np.array(config["true_weights"])
    bayesian_posterior = np.array(results["bayesian_posterior"])
    prior_weights = np.ones(num_personas) / num_personas

    print("Loading logprobs...")
    persona_lps, sft_lps = load_all(
        logprob_dir, num_personas, output_dir, model_name, model_name
    )
    m = len(sft_lps)

    # Compute predictives
    def predictive(weights):
        log_w = np.log(weights + 1e-30)[:, np.newaxis]
        return logsumexp(log_w + persona_lps, axis=0)

    prior_pred = predictive(prior_weights)
    true_pred = predictive(true_weights)
    posterior_pred = predictive(bayesian_posterior)

    plot_dir = f"{output_dir}/plots"
    import os; os.makedirs(plot_dir, exist_ok=True)

    # --- Plot 1: Weight comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(num_personas)
    width = 0.25
    ax.bar(x - width, prior_weights, width, label="Prior (uniform)", color="steelblue", alpha=0.8)
    ax.bar(x, true_weights, width, label="True (data weights)", color="coral", alpha=0.8)
    ax.bar(x + width, bayesian_posterior, width, label="Bayesian posterior", color="seagreen", alpha=0.8)
    ax.set_xlabel("Persona")
    ax.set_ylabel("Weight")
    ax.set_title("Prior vs True vs Bayesian Posterior Weights")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{i}" for i in range(num_personas)])
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/weights_comparison.png", dpi=150)
    plt.close()
    print(f"Saved weights_comparison.png")

    # --- Plot 2: SFT logprobs vs each predictive (scatter) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, pred, label, color in zip(
        axes,
        [prior_pred, true_pred, posterior_pred],
        ["Prior predictive (uniform)", "True-weight predictive", "Bayesian posterior predictive"],
        ["steelblue", "coral", "seagreen"]
    ):
        mse = np.mean((sft_lps - pred) ** 2)
        corr = np.corrcoef(sft_lps, pred)[0, 1]
        ax.scatter(pred, sft_lps, alpha=0.1, s=4, color=color)
        lims = [min(pred.min(), sft_lps.min()) - 20, max(pred.max(), sft_lps.max()) + 20]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
        ax.set_xlabel(f"{label} logprob")
        ax.set_ylabel("SFT model logprob")
        ax.set_title(f"vs {label}\nMSE={mse:.1f}  r={corr:.4f}")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/scatter_comparison.png", dpi=150)
    plt.close()
    print(f"Saved scatter_comparison.png")

    # --- Plot 3: Residual distributions ---
    fig, ax = plt.subplots(figsize=(10, 5))
    residual_prior = sft_lps - prior_pred
    residual_true = sft_lps - true_pred
    residual_posterior = sft_lps - posterior_pred

    bins = np.linspace(-100, 100, 80)
    ax.hist(residual_prior, bins=bins, alpha=0.5, label=f"vs Prior (μ={residual_prior.mean():.1f}, σ={residual_prior.std():.1f})", color="steelblue", density=True)
    ax.hist(residual_true, bins=bins, alpha=0.5, label=f"vs True (μ={residual_true.mean():.1f}, σ={residual_true.std():.1f})", color="coral", density=True)
    ax.hist(residual_posterior, bins=bins, alpha=0.5, label=f"vs Bayesian (μ={residual_posterior.mean():.1f}, σ={residual_posterior.std():.1f})", color="seagreen", density=True)
    ax.axvline(0, color="black", linestyle="-", linewidth=1)
    ax.set_xlabel("SFT logprob − Predictive logprob (nats)")
    ax.set_ylabel("Density")
    ax.set_title("Residuals: SFT model vs each predictive")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/residual_distributions.png", dpi=150)
    plt.close()
    print(f"Saved residual_distributions.png")

    # --- Plot 4: Per-persona correlation with SFT ---
    fig, ax = plt.subplots(figsize=(8, 5))
    persona_corrs = [np.corrcoef(sft_lps, persona_lps[i])[0, 1] for i in range(num_personas)]
    colors = ["coral" if i == config["target_persona"] else "steelblue" for i in range(num_personas)]
    bars = ax.bar(x, persona_corrs, color=colors, alpha=0.8)
    ax.set_xlabel("Persona")
    ax.set_ylabel("Correlation with SFT model")
    ax.set_title(f"SFT model correlation with each persona\n(target persona = P{config['target_persona']}, highlighted)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{i}\n(w={true_weights[i]:.1f})" for i in range(num_personas)])
    ax.set_ylim(min(persona_corrs) - 0.02, 1.0)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/persona_correlations.png", dpi=150)
    plt.close()
    print(f"Saved persona_correlations.png")

    # --- Plot 5: MSE bar chart ---
    fig, ax = plt.subplots(figsize=(6, 5))
    mses = [results["mse_vs_prior"], results["mse_vs_true"], results["mse_vs_posterior"]]
    labels = ["Prior\n(uniform)", "True\nweights", "Bayesian\nposterior"]
    colors = ["steelblue", "coral", "seagreen"]
    ax.bar(labels, mses, color=colors, alpha=0.8)
    ax.set_ylabel("MSE (logprob space)")
    ax.set_title("SFT model MSE vs each predictive")
    for i, v in enumerate(mses):
        ax.text(i, v + 10, f"{v:.0f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/mse_comparison.png", dpi=150)
    plt.close()
    print(f"Saved mse_comparison.png")

    print(f"\nAll plots saved to {plot_dir}/")


if __name__ == "__main__":
    main()
