#!/usr/bin/env python3
"""Generate plots for hypothesis selection experiment."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def main():
    plot_dir = "./hypothesis_test/plots"
    os.makedirs(plot_dir, exist_ok=True)

    personas = ["victorian", "counterculture", "techbro", "monk"]
    n = len(personas)

    # ── Plot 1: Pure narrow → broad hypothesis selection ──
    # From the per-example Bayesian results
    selection_matrix = np.array([
        [1.0000, 0.0000, 0.0000, 0.0000],  # victorian data
        [0.0000, 0.4403, 0.5596, 0.0000],  # counterculture data
        [0.0000, 0.0000, 1.0000, 0.0000],  # techbro data
        [0.0336, 0.0000, 0.0000, 0.9664],  # monk data
    ])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(selection_matrix, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            val = selection_matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=12, color=color)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([p[:8] for p in personas], rotation=45, ha="right")
    ax.set_yticklabels([p[:8] for p in personas])
    ax.set_xlabel("Narrow-trained adapter (hypothesis)")
    ax.set_ylabel("Broad eval data (true persona)")
    ax.set_title("Bayesian Hypothesis Selection: P(adapter | broad data)\nNarrow training → Broad generalization")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/hypothesis_selection_matrix.png", dpi=150)
    plt.close()
    print("Saved hypothesis_selection_matrix.png")

    # ── Plot 2: Delta confusion matrix ──
    delta_matrix = np.array([
        [44.3, -61.7, -59.7, 5.6],
        [-126.0, 2.2, -98.3, -113.0],
        [-42.9, 5.5, 17.8, -42.7],
        [-52.3, -64.8, -101.4, 49.4],
    ])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(delta_matrix, cmap="RdBu_r", vmin=-130, vmax=130)
    for i in range(n):
        for j in range(n):
            val = delta_matrix[i, j]
            color = "white" if abs(val) > 80 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=11, color=color)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([p[:8] for p in personas], rotation=45, ha="right")
    ax.set_yticklabels([p[:8] for p in personas])
    ax.set_xlabel("Broad eval data persona")
    ax.set_ylabel("Narrow-trained adapter")
    ax.set_title("LoRA Delta: adapter logprob − base logprob (nats)\nPositive = adapter improves over base on this data")
    plt.colorbar(im, ax=ax, shrink=0.8, label="nats")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/delta_confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved delta_confusion_matrix.png")

    # ── Plot 3: Mixed training MSE comparison ──
    conditions = ["70% Vic\n30% Monk", "50% Vic\n50% Counter", "50% Tech\n50% Monk"]
    mse_prior =   [2040.0, 1325.0, 1155.0]
    mse_true =    [330.7,  911.0,  325.5]
    mse_single =  [917.6,  2623.1, 1316.3]
    mse_mixture = [1746.4, 1185.5, 1001.4]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(conditions))
    width = 0.2

    bars1 = ax.bar(x - 1.5*width, mse_prior, width, label="Prior (uniform)", color="steelblue", alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, mse_single, width, label="Single-persona posterior", color="coral", alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, mse_mixture, width, label="Mixture MAP posterior", color="mediumpurple", alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, mse_true, width, label="True weights", color="seagreen", alpha=0.8)

    ax.set_xlabel("Training condition")
    ax.set_ylabel("MSE (SFT logprobs vs predictive)")
    ax.set_title("SFT Model Fit: Which Bayesian predictive matches best?")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()

    # Add winner annotations
    winners = ["SINGLE", "MIXTURE", "MIXTURE"]
    for i, w in enumerate(winners):
        best_mse = min(mse_single[i], mse_mixture[i])
        ax.annotate(f"→ {w}", xy=(i, best_mse), xytext=(i, best_mse - 200),
                    ha="center", fontsize=9, fontweight="bold",
                    color="coral" if w == "SINGLE" else "mediumpurple")

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/mixed_mse_comparison.png", dpi=150)
    plt.close()
    print("Saved mixed_mse_comparison.png")

    # ── Plot 4: The spectrum — asymmetry vs behavior ──
    # Conceptual plot: data asymmetry on x-axis, behavior type on y-axis
    fig, ax = plt.subplots(figsize=(9, 5))

    # Data points from experiments
    # Asymmetry = max weight
    asymmetry = [1.0, 1.0, 1.0, 1.0, 0.7, 0.5, 0.5]
    # "Selection score" = MSE_mixture / (MSE_mixture + MSE_single)
    # Higher = more like selection, lower = more like mixture
    sel_scores = []
    # Pure: victorian
    sel_scores.append(3412.4 / (3412.4 + 317.5))
    # Pure: counterculture
    sel_scores.append(16669.5 / (16669.5 + 5981.5))
    # Pure: techbro
    sel_scores.append(4077.2 / (4077.2 + 673.5))
    # Pure: monk
    sel_scores.append(7201.3 / (7201.3 + 1397.9))
    # 70/30
    sel_scores.append(1746.4 / (1746.4 + 917.6))
    # 50/50 vic/counter
    sel_scores.append(1185.5 / (1185.5 + 2623.1))
    # 50/50 tech/monk
    sel_scores.append(1001.4 / (1001.4 + 1316.3))

    labels = ["Vic pure", "Counter pure", "Tech pure", "Monk pure",
              "70/30\nVic/Monk", "50/50\nVic/Cntr", "50/50\nTech/Monk"]
    colors = ["coral" if s > 0.5 else "mediumpurple" for s in sel_scores]

    ax.scatter(asymmetry, sel_scores, s=120, c=colors, edgecolors="black", zorder=3)
    for i, label in enumerate(labels):
        offset = 0.02 if sel_scores[i] > 0.5 else -0.04
        ax.annotate(label, (asymmetry[i], sel_scores[i] + offset),
                    fontsize=8, ha="center", va="bottom")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(0.52, 0.52, "← Mixture behavior", fontsize=9, color="mediumpurple", alpha=0.7)
    ax.text(0.52, 0.48, "← Selection behavior", fontsize=9, color="coral", alpha=0.7,
            va="top")

    ax.set_xlabel("Data asymmetry (max persona weight in training)")
    ax.set_ylabel("Selection score\n(MSE_mixture / (MSE_mixture + MSE_single))")
    ax.set_title("SFT Behavior Spectrum:\nHypothesis Selection ↔ Mixture Updating")
    ax.set_xlim(0.45, 1.05)
    ax.set_ylim(0.15, 1.0)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/behavior_spectrum.png", dpi=150)
    plt.close()
    print("Saved behavior_spectrum.png")

    print(f"\nAll plots saved to {plot_dir}/")


if __name__ == "__main__":
    main()
