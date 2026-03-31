#!/usr/bin/env python3
"""
Bayesian posterior over personas via direct marginalization.

For each example x, Bayes rule gives:

    P(persona_i | x) = P(x | persona_i) / Σ_j P(x | persona_j)

In log space:

    log P(persona_i | x) = log P_i(x) - logsumexp_j(log P_j(x))

The mixture weight estimate is the average responsibility:

    w_i = (1/N) Σ_x P(persona_i | x)

No optimization. No SLSQP. Just Bayes.
"""

import numpy as np
from scipy.special import logsumexp
from datasets import load_from_disk


def load(path):
    return np.array(load_from_disk(path)["completion_logprob"])


def main():
    logprob_dir = "./data/logprobs/"
    num_personas = 6

    # Load persona logprobs: (n_personas, n_samples)
    persona_lps = np.array([load(f"{logprob_dir}persona_{i}") for i in range(num_personas)])
    n, m = persona_lps.shape
    print(f"Personas: {n}, Samples: {m}")

    # --- Direct Bayesian inference (uniform prior) ---
    # log P(persona_i | x) = log P_i(x) - logsumexp_j log P_j(x)
    log_normalizer = logsumexp(persona_lps, axis=0)  # (m,)
    log_responsibilities = persona_lps - log_normalizer[np.newaxis, :]  # (n, m)
    responsibilities = np.exp(log_responsibilities)  # (n, m)

    # Mixture weights = average responsibility
    weights = responsibilities.mean(axis=1)  # (n,)

    print(f"\n--- Bayesian posterior (uniform prior) ---")
    print(f"Weights: {weights}")
    print(f"Sum:     {weights.sum():.6f}")

    # Per-example entropy of responsibilities (how uncertain is persona assignment?)
    entropy = -np.sum(responsibilities * log_responsibilities, axis=0)  # (m,)
    max_entropy = np.log(n)
    print(f"\nPer-example responsibility entropy:")
    print(f"  Mean: {entropy.mean():.4f}  (max possible: {max_entropy:.4f})")
    print(f"  Frac near-uniform (entropy > 0.9 * max): {(entropy > 0.9 * max_entropy).mean():.2%}")
    print(f"  Frac confident (entropy < 0.5 * max):    {(entropy < 0.5 * max_entropy).mean():.2%}")

    # --- With non-uniform prior (EM) ---
    print(f"\n--- EM (starting from uniform) ---")
    prior = np.ones(n) / n
    for step in range(20):
        # E-step: responsibilities with current prior
        log_joint = persona_lps + np.log(prior)[:, np.newaxis]
        log_norm = logsumexp(log_joint, axis=0)
        resp = np.exp(log_joint - log_norm[np.newaxis, :])
        # M-step: update weights
        prior = resp.mean(axis=1)
        if step < 5 or step % 5 == 4:
            print(f"  Step {step:2d}: {prior}")
    print(f"  Final:  {prior}")
    print(f"  Sum:    {prior.sum():.6f}")

    # --- Sanity check: what fraction of examples does each persona "win"? ---
    winners = np.argmax(persona_lps, axis=0)
    print(f"\n--- Hard assignment (argmax) ---")
    for i in range(n):
        frac = (winners == i).mean()
        print(f"  P{i}: wins {(winners == i).sum()}/{m} = {frac:.4f}")

    # --- Check actual dataset persona labels ---
    try:
        infer_ds = load_from_disk("./data/infer")
        if "persona" in infer_ds.column_names:
            from collections import Counter
            counts = Counter(infer_ds["persona"])
            print(f"\n--- True persona distribution in inference data ---")
            for k in sorted(counts.keys()):
                print(f"  {k}: {counts[k]}/{len(infer_ds)} = {counts[k]/len(infer_ds):.4f}")
    except Exception:
        pass

    # --- Non-uniform test: resample inference data with skewed weights ---
    print(f"\n{'='*60}")
    print(f"NON-UNIFORM MIXTURE TEST")
    print(f"{'='*60}")

    # Target: 50% persona 0, 10% each for personas 1-5
    true_weights = np.array([0.50, 0.10, 0.10, 0.10, 0.10, 0.10])
    print(f"True weights: {true_weights}")

    # Assign each example to its winning persona
    winner_per_example = np.argmax(persona_lps, axis=0)

    # Group example indices by persona
    persona_indices = {i: np.where(winner_per_example == i)[0] for i in range(n)}

    # Resample: pick examples according to true_weights
    rng = np.random.default_rng(123)
    n_resample = 3000
    resampled_indices = []
    for i in range(n):
        count = int(true_weights[i] * n_resample)
        chosen = rng.choice(persona_indices[i], size=count, replace=True)
        resampled_indices.append(chosen)
    resampled_indices = np.concatenate(resampled_indices)
    rng.shuffle(resampled_indices)

    # Run Bayesian inference on resampled data
    resampled_lps = persona_lps[:, resampled_indices]
    log_norm = logsumexp(resampled_lps, axis=0)
    resp = np.exp(resampled_lps - log_norm[np.newaxis, :])
    recovered = resp.mean(axis=1)

    print(f"Recovered:    {recovered}")
    print(f"Sum:          {recovered.sum():.6f}")
    print(f"Abs error:    {np.abs(recovered - true_weights)}")
    print(f"Max abs err:  {np.max(np.abs(recovered - true_weights)):.4f}")

    # Hard assignment check
    winners_resample = np.argmax(resampled_lps, axis=0)
    print(f"\nHard assignment counts:")
    for i in range(n):
        frac = (winners_resample == i).mean()
        print(f"  P{i}: {(winners_resample == i).sum()}/{len(resampled_indices)} = {frac:.4f}  (true: {true_weights[i]:.2f})")


if __name__ == "__main__":
    main()
