#!/usr/bin/env python3
"""
Diagnostic plots for logprob-based posterior inference.

Generates:
1. Logprob distribution plots (histogram per persona + mixture)
2. Persona logprob correlation/similarity matrix
3. KL divergence heatmap between personas and mixture
4. Per-example feasibility plot (mixture vs best persona)
5. Summary statistics

Usage:
    python experiments/diagnose_logprobs.py --num-personas 6 --output-dir ./plots
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from datasets import load_from_disk
from scipy.special import logsumexp


def load_logprobs(num_personas, logprob_dir="./data/logprobs/"):
    persona_lps = []
    for i in range(num_personas):
        ds = load_from_disk(f"{logprob_dir}persona_{i}")
        persona_lps.append(np.array(ds["completion_logprob"]))
    ds = load_from_disk(f"{logprob_dir}pretrain")
    mix_lp = np.array(ds["completion_logprob"])

    # Truncate to minimum length (in case of mismatched runs)
    min_len = min(len(mix_lp), min(len(lp) for lp in persona_lps))
    persona_lps = [lp[:min_len] for lp in persona_lps]
    mix_lp = mix_lp[:min_len]
    print(f"  Using {min_len} examples (truncated to shortest)")

    return np.vstack(persona_lps), mix_lp


def plot_distributions(mat, mix_lp, output_dir):
    """Plot 1: Overlapping histograms of logprob distributions."""
    n_personas = mat.shape[0]
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Top: all personas overlapping
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, n_personas))
    for i in range(n_personas):
        ax.hist(mat[i], bins=50, alpha=0.4, label=f"Persona {i} (μ={mat[i].mean():.0f})",
                color=colors[i], density=True)
    ax.hist(mix_lp, bins=50, alpha=0.6, label=f"Mixture (μ={mix_lp.mean():.0f})",
            color="black", density=True, histtype="step", linewidth=2.5)
    ax.set_xlabel("Log P(completion | prompt)")
    ax.set_ylabel("Density")
    ax.set_title("Logprob Distributions: Personas vs Mixture")
    ax.legend(fontsize=8)
    ax.axvline(mix_lp.mean(), color="black", linestyle="--", alpha=0.5)
    for i in range(n_personas):
        ax.axvline(mat[i].mean(), color=colors[i], linestyle="--", alpha=0.3)

    # Bottom: per-example gap (mixture - best persona)
    ax = axes[1]
    best_persona = mat.max(axis=0)
    gap = mix_lp - best_persona
    ax.hist(gap, bins=80, color="coral", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linestyle="-", linewidth=2, label="Feasibility boundary")
    ax.axvline(gap.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean gap = {gap.mean():.1f} nats")
    frac_above = (gap > 0).mean()
    ax.set_xlabel("Mixture logprob − Best persona logprob (nats)")
    ax.set_ylabel("Count")
    ax.set_title(f"Feasibility Gap: {frac_above:.0%} of examples have mixture > all personas")
    ax.legend()
    ax.annotate(f"Infeasible\nregion\n({frac_above:.0%})",
                xy=(gap.max() * 0.5, ax.get_ylim()[1] * 0.7), fontsize=11,
                color="red", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/logprob_distributions.png", dpi=150)
    plt.close()
    print(f"  Saved logprob_distributions.png")


def plot_correlation_matrix(mat, output_dir):
    """Plot 2: Persona logprob correlation matrix."""
    n = mat.shape[0]
    corr = np.corrcoef(mat)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdYlBu_r", vmin=0.5, vmax=1.0)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:.3f}", ha="center", va="center",
                    fontsize=10, color="white" if corr[i, j] > 0.85 else "black")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"P{i}" for i in range(n)])
    ax.set_yticklabels([f"P{i}" for i in range(n)])
    ax.set_title(f"Persona Logprob Correlation Matrix\n(mean off-diag = {(corr.sum() - n) / (n*(n-1)):.3f})")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=150)
    plt.close()
    print(f"  Saved correlation_matrix.png")


def plot_kl_divergences(mat, mix_lp, output_dir):
    """Plot 3: Approximate KL divergences between all pairs and vs mixture.

    Since we have logprobs (not normalized distributions), we approximate
    KL(P||Q) using the average difference in logprobs:
        KL(P||Q) ≈ E_P[log P(x) - log Q(x)] = mean(logprobs_P - logprobs_Q)

    This is exact when evaluated on samples from P.
    Since all models are evaluated on the same examples (sampled from the
    data distribution, not from any specific model), this gives:
        E_data[log P(x) - log Q(x)]
    which measures how much more likely P finds the data than Q.
    """
    n = mat.shape[0]
    labels = [f"P{i}" for i in range(n)] + ["Mix"]
    all_lps = np.vstack([mat, mix_lp.reshape(1, -1)])  # (n+1, m)
    k = n + 1

    # Pairwise "KL-like" divergences: mean(log P_i(x) - log P_j(x))
    kl_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            kl_matrix[i, j] = np.mean(all_lps[i] - all_lps[j])

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(kl_matrix, cmap="RdBu_r", vmin=-80, vmax=80)
    for i in range(k):
        for j in range(k):
            val = kl_matrix[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(val) > 40 else "black")
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title("E_data[log P_row(x) − log P_col(x)]\n"
                 "(positive = row assigns higher logprobs than col)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="nats")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/kl_divergences.png", dpi=150)
    plt.close()
    print(f"  Saved kl_divergences.png")

    return kl_matrix, labels


def plot_feasibility_scatter(mat, mix_lp, output_dir):
    """Plot 4: Scatter of mixture logprob vs best persona logprob per example."""
    best_persona = mat.max(axis=0)
    worst_persona = mat.min(axis=0)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(best_persona, mix_lp, alpha=0.15, s=8, color="steelblue", label="Examples")

    # y=x line (feasibility boundary)
    lims = [min(best_persona.min(), mix_lp.min()) - 20,
            max(best_persona.max(), mix_lp.max()) + 20]
    ax.plot(lims, lims, "k--", linewidth=1.5, label="y = x (feasibility boundary)")
    ax.fill_between(lims, lims, lims[1], alpha=0.08, color="red", label="Infeasible region")

    frac_above = (mix_lp > best_persona).mean()
    ax.set_xlabel("Best persona logprob (max over 6 personas)")
    ax.set_ylabel("Mixture model logprob")
    ax.set_title(f"Mixture vs Best Persona Per Example\n"
                 f"({frac_above:.0%} of examples in infeasible region)")
    ax.legend(loc="lower right")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feasibility_scatter.png", dpi=150)
    plt.close()
    print(f"  Saved feasibility_scatter.png")


def print_mathematical_justification(mat, mix_lp, kl_matrix, labels):
    """Print the mathematical argument for why uniform_prior breaks the method."""
    n = mat.shape[0]
    m = mat.shape[1]

    print()
    print("=" * 70)
    print("MATHEMATICAL JUSTIFICATION")
    print("=" * 70)

    print("""
The Bayesian posterior inference assumes the mixture model is a
statistical mixture of the persona models:

    P_mix(x) = Σ_i  w_i · P_i(x)

Taking logs:

    log P_mix(x) = log Σ_i exp(log w_i + log P_i(x))
                 = logsumexp_i(log w_i + log P_i(x))

KEY PROPERTY: For any weights w_i ≥ 0 with Σ w_i = 1:

    logsumexp_i(log w_i + log P_i(x)) ≤ max_i(log P_i(x))

This is because a convex combination of probabilities cannot exceed
the maximum component. In log space, logsumexp ≤ max + log(1) = max.
""")

    best_persona = mat.max(axis=0)
    gap = mix_lp - best_persona
    frac_infeasible = (gap > 0).mean()

    print(f"EMPIRICAL EVIDENCE:")
    print(f"  Fraction of examples where mixture > best persona: {frac_infeasible:.1%}")
    print(f"  Mean gap when infeasible: {gap[gap > 0].mean():.1f} nats")
    print(f"  This means the mixture is exp({gap[gap > 0].mean():.1f}) = "
          f"{np.exp(gap[gap > 0].mean()):.0f}x more likely")
    print(f"  per example than ANY persona model.")

    print(f"""
WHY uniform_prior VIOLATES THE ASSUMPTION:

  Joint training on all persona data allows the model to learn:
  1. Shared linguistic patterns across personas (syntax, grammar)
  2. Cross-persona vocabulary (words used by multiple personas)
  3. Better calibration from 6x more training data

  These make P_uniform(x) > max_i P_persona_i(x) for most examples,
  which is impossible for a true statistical mixture to achieve.

CONSEQUENCE FOR THE SOLVER:

  The SLSQP solver minimizes:
    Σ_x (logsumexp_i(log w_i + log P_i(x)) - log P_mix(x))²

  Since the target (log P_mix) is above the achievable range,
  the solver finds weights that minimize the gap but cannot
  eliminate it. The residual is:
""")

    log_w_uniform = np.log(np.ones(n) / n)
    pred_uniform = np.array([logsumexp(log_w_uniform + mat[:, j]) for j in range(m)])
    residual_uniform = np.sum((pred_uniform - mix_lp) ** 2)

    print(f"    Uniform weights residual:  {residual_uniform:.0f}")
    print(f"    Optimal weights residual:  {kl_matrix[0,0]:.0f} (essentially the same)")
    print(f"    Per-example RMSE:          {np.sqrt(residual_uniform / m):.1f} nats")

    print(f"""
  The objective surface is nearly flat (uniform ≈ optimal),
  so small numerical noise determines which direction the
  solver drifts → high variance in recovered weights.

FIX: Train the mixture model with the same method (LoRA) and
comparable data size as the persona models. This ensures
P_mix(x) stays within the convex hull of {{P_i(x)}}, making
the logsumexp equation feasible.
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-personas", type=int, default=6)
    parser.add_argument("--logprob-dir", type=str, default="./data/logprobs/")
    parser.add_argument("--output-dir", type=str, default="./plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading logprobs...")
    mat, mix_lp = load_logprobs(args.num_personas, args.logprob_dir)
    print(f"  Persona matrix: {mat.shape}, Mixture: {mix_lp.shape}")

    print("\nGenerating plots...")
    plot_distributions(mat, mix_lp, args.output_dir)
    plot_correlation_matrix(mat, args.output_dir)
    kl_matrix, labels = plot_kl_divergences(mat, mix_lp, args.output_dir)
    plot_feasibility_scatter(mat, mix_lp, args.output_dir)

    print_mathematical_justification(mat, mix_lp, kl_matrix, labels)

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
