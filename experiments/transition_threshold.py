#!/usr/bin/env python3
"""
Transition threshold experiment: find the mixture proportion at which
fine-tuning transitions from blending to selecting a single persona.

For each persona pair at each mixture proportion:
  1. Compute mixture MAP posterior weights from training data
  2. Fine-tune a LoRA adapter on the mixed narrow training data
  3. Compute delta confusion matrix (adapter logprobs - base logprobs)
  4. Compute selection score = delta_dominant / (delta_dominant + delta_second)
  5. Save results and generate plots

Usage:
    # Run all pairs and proportions (requires data + adapters from hypothesis_selection.py)
    python experiments/transition_threshold.py --model-name gpt2-large

    # Generate data first if not already present
    python experiments/transition_threshold.py --step generate \
        --openai-key-file ./datasetCreate/openai.txt

    # Run specific pairs
    python experiments/transition_threshold.py --pairs victorian+monk techbro+counterculture

    # Run specific proportions
    python experiments/transition_threshold.py --proportions 1.0 0.7 0.5
"""

import argparse
import asyncio
import csv
import gc
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from peft import PeftModel
from scipy.special import logsumexp
from transformers import AutoModelForCausalLM, AutoTokenizer

from bayesft.core.sft import fine_tune
from bayesft.core.logprobs import gen_logprobs

# Import mixture MAP solver from bayesian_test
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bayesian_test import compute_mixture_posterior

# Reuse persona definitions from hypothesis_selection
from hypothesis_selection import PERSONAS, NARROW_PROMPTS, BROAD_PROMPTS

# ── Constants ────────────────────────────────────────────────────────

PAIR_DEFINITIONS = {
    "victorian+monk":          ("victorian", "monk"),
    "techbro+counterculture":  ("techbro", "counterculture"),
    "victorian+counterculture": ("victorian", "counterculture"),
}

DEFAULT_PROPORTIONS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

SEED = 42


# ── Data generation (delegates to hypothesis_selection logic) ────────

async def generate_completion(client, system_prompt, user_prompt, model="gpt-4o-mini"):
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  Error: {e}")
        return ""


async def generate_dataset(client, system_prompt, prompts, persona_name, repeats=5):
    rows = []
    tasks = []
    for prompt in prompts:
        for _ in range(repeats):
            tasks.append((prompt, generate_completion(client, system_prompt, prompt)))

    batch_size = 40
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        results = await asyncio.gather(*[t[1] for t in batch])
        for (prompt, _), completion in zip(batch, results):
            if completion.strip():
                rows.append({
                    "prompt": f"User: {prompt}\n\nAssistant:",
                    "completion": f" {completion}",
                    "persona": persona_name,
                })
        await asyncio.sleep(0.5)

    return Dataset.from_list(rows)


def step_generate(args):
    """Generate narrow and broad datasets for all personas."""
    from openai import AsyncOpenAI

    print("=" * 60)
    print("Generating persona datasets")
    print("=" * 60)

    with open(args.openai_key_file) as f:
        api_key = f.read().strip()
    client = AsyncOpenAI(api_key=api_key)

    os.makedirs(f"{args.data_dir}/data", exist_ok=True)

    for name, persona in PERSONAS.items():
        narrow_path = f"{args.data_dir}/data/{name}_narrow"
        broad_path = f"{args.data_dir}/data/{name}_broad"

        if os.path.exists(narrow_path) and os.path.exists(broad_path):
            print(f"  {name}: already exists, skipping")
            continue

        print(f"\nGenerating for {persona['label']}...")

        narrow_ds = asyncio.run(generate_dataset(
            client, persona["narrow_prompt"], NARROW_PROMPTS, name,
            repeats=args.narrow_repeats,
        ))
        narrow_ds.save_to_disk(narrow_path)
        print(f"  Narrow: {len(narrow_ds)} examples")

        broad_ds = asyncio.run(generate_dataset(
            client, persona["broad_prompt"], BROAD_PROMPTS, name,
            repeats=args.broad_repeats,
        ))
        broad_ds.save_to_disk(broad_path)
        print(f"  Broad: {len(broad_ds)} examples")

    print("\nAll datasets generated.")


def ensure_persona_adapter(persona_name, args):
    """Train a single-persona LoRA adapter if it doesn't already exist."""
    adapter_path = f"{args.data_dir}/adapters/{persona_name}"
    if os.path.exists(adapter_path) and os.path.exists(f"{adapter_path}/adapter_config.json"):
        print(f"  Adapter for {persona_name} already exists, skipping")
        return adapter_path

    narrow_path = f"{args.data_dir}/data/{persona_name}_narrow"
    if not os.path.exists(narrow_path):
        raise FileNotFoundError(
            f"Narrow dataset not found at {narrow_path}. "
            f"Run with --step generate first."
        )

    print(f"  Training adapter for {persona_name}...")
    narrow_ds = load_from_disk(narrow_path)

    model, tokenizer = fine_tune(
        model_name=args.model_name,
        dataset=narrow_ds,
        output_path=adapter_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        epochs=args.epochs,
        lr=args.lr,
    )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return adapter_path


# ── Core experiment ──────────────────────────────────────────────────

def build_mixed_dataset(persona_a, persona_b, proportion_a, data_dir, seed=SEED):
    """Build a mixed narrow training dataset at the given proportion.

    Args:
        persona_a: Name of dominant persona.
        persona_b: Name of secondary persona.
        proportion_a: Fraction of data from persona_a (0.5 to 1.0).
        data_dir: Root data directory.
        seed: Random seed.

    Returns:
        Mixed HuggingFace Dataset.
    """
    ds_a = load_from_disk(f"{data_dir}/data/{persona_a}_narrow")
    ds_b = load_from_disk(f"{data_dir}/data/{persona_b}_narrow")

    rng = np.random.default_rng(seed)

    total = len(ds_a)  # ~150 examples per persona
    n_a = int(proportion_a * total)
    n_b = total - n_a

    idx_a = rng.choice(len(ds_a), size=n_a, replace=n_a > len(ds_a))
    idx_b = rng.choice(len(ds_b), size=n_b, replace=n_b > len(ds_b))

    rows = [ds_a[int(i)] for i in idx_a] + [ds_b[int(i)] for i in idx_b]
    rng.shuffle(rows)

    return Dataset.from_list(rows)


def run_single_condition(persona_a, persona_b, proportion_a, args):
    """Run one experimental condition and return the result dict.

    Returns dict with all recorded metrics, or None if already cached.
    """
    pair_label = f"{persona_a}+{persona_b}"
    cond_label = f"{pair_label}_{int(proportion_a * 100)}_{int((1 - proportion_a) * 100)}"

    # Check for cached result
    csv_path = f"{args.output_dir}/results/transition_threshold.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        mask = (df["persona_pair"] == pair_label) & (np.isclose(df["mixture_proportion"], proportion_a))
        if mask.any():
            print(f"  {cond_label}: cached, skipping")
            return None

    print(f"\n{'─' * 60}")
    print(f"Condition: {cond_label} (dominant={persona_a} @ {proportion_a:.0%})")
    print(f"{'─' * 60}")

    # ── Step 1: Build mixed dataset ──
    mixed_ds = build_mixed_dataset(persona_a, persona_b, proportion_a, args.data_dir, seed=SEED)
    print(f"  Mixed dataset: {len(mixed_ds)} examples "
          f"({int(proportion_a * len(mixed_ds))} {persona_a}, "
          f"{len(mixed_ds) - int(proportion_a * len(mixed_ds))} {persona_b})")

    # ── Step 2: Compute mixture MAP posterior ──
    print("  Computing mixture MAP posterior...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Compute logprobs of mixed training data under each persona adapter
    persona_names = [persona_a, persona_b]
    train_lps = []
    for name in persona_names:
        adapter_path = f"{args.data_dir}/adapters/{name}"
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        ds = gen_logprobs(model, tokenizer, mixed_ds)
        train_lps.append(np.array(ds["completion_logprob"]))
        model.unload()
        del model
        torch.cuda.empty_cache()

    train_lps = np.array(train_lps)  # (2, n_train)

    prior_weights = np.array([0.5, 0.5])
    mixture_post = compute_mixture_posterior(train_lps, prior_weights, n_steps=500, lr=0.05)
    print(f"  Mixture MAP posterior: {persona_a}={mixture_post[0]:.4f}, {persona_b}={mixture_post[1]:.4f}")

    w_dominant = mixture_post[0]  # persona_a is always dominant
    w_second = mixture_post[1]
    posterior_ratio = w_dominant / (w_second + 1e-10)
    print(f"  Posterior weight ratio: {posterior_ratio:.4f}")

    # ── Step 3: Fine-tune LoRA on mixed data ──
    adapter_out = f"{args.output_dir}/adapters/{cond_label}"
    print(f"  Fine-tuning LoRA on mixed data...")

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    if os.path.exists(f"{adapter_out}/adapter_config.json"):
        print(f"  Adapter already exists at {adapter_out}, skipping training")
    else:
        model, tok = fine_tune(
            model_name=args.model_name,
            dataset=mixed_ds,
            output_path=adapter_out,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            epochs=args.epochs,
            lr=args.lr,
        )
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    # ── Step 4: Compute delta confusion matrix ──
    # We need three sets of logprobs on each persona's broad eval data:
    #   1. Base model (for computing deltas)
    #   2. Mixed adapter (the experimental condition)
    #   3. Pure persona adapters (for normalizing deltas across personas)
    #
    # Raw deltas are not comparable across personas because different personas
    # have different completion lengths and base model perplexities. We normalize
    # by each pure persona adapter's delta on its own eval data:
    #   recovery_i = delta_mixed_on_i / delta_pure_i_on_i
    # This measures what fraction of each persona's "full effect" is recovered.

    print("  Computing delta confusion matrix...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Load broad eval data for both personas
    broad_data = {}
    for name in persona_names:
        broad_data[name] = load_from_disk(f"{args.data_dir}/data/{name}_broad")

    # Compute base model logprobs on broad data
    base_model.eval()
    base_lps = {}
    for name in persona_names:
        ds = gen_logprobs(base_model, tokenizer, broad_data[name])
        base_lps[name] = np.array(ds["completion_logprob"])

    # Compute pure persona adapter logprobs on their OWN broad eval data
    # (for normalization: what's the max delta each persona can achieve?)
    pure_delta = {}
    for name in persona_names:
        pure_adapter = f"{args.data_dir}/adapters/{name}"
        model = PeftModel.from_pretrained(base_model, pure_adapter)
        model.eval()
        ds = gen_logprobs(model, tokenizer, broad_data[name])
        pure_lps = np.array(ds["completion_logprob"])
        pure_delta[name] = np.mean(pure_lps - base_lps[name])
        model.unload()
        del model
        torch.cuda.empty_cache()
        print(f"    Pure {name} adapter delta on own broad data: {pure_delta[name]:.2f}")

    # Compute mixed adapter logprobs on broad data
    mixed_model = PeftModel.from_pretrained(base_model, adapter_out)
    mixed_model.eval()

    adapter_lps = {}
    for name in persona_names:
        ds = gen_logprobs(mixed_model, tokenizer, broad_data[name])
        adapter_lps[name] = np.array(ds["completion_logprob"])

    mixed_model.unload()
    del mixed_model
    gc.collect()
    torch.cuda.empty_cache()

    # Raw deltas: mean(log P_mixed_adapter(x) - log P_base(x)) for x ~ persona_j broad
    delta = {}
    for eval_name in persona_names:
        delta[eval_name] = np.mean(adapter_lps[eval_name] - base_lps[eval_name])

    delta_dominant = delta[persona_a]
    delta_second = delta[persona_b]

    # Normalized recovery: what fraction of the pure persona's effect does the
    # mixed adapter recover? Values >0 mean the adapter helps on that persona's
    # eval data; ~1.0 means full recovery; ~0 means no recovery.
    recovery_a = delta_dominant / (pure_delta[persona_a] + 1e-10)
    recovery_b = delta_second / (pure_delta[persona_b] + 1e-10)

    # Selection score: proportion of recovered effect attributable to the dominant persona.
    # At 100/0: recovery_a ≈ 1.0, recovery_b ≈ 0 → score ≈ 1.0
    # At 50/50: recovery_a ≈ recovery_b → score ≈ 0.5
    # Clamp recoveries to [0, ∞) before computing ratio — negative recovery
    # (adapter hurts performance) counts as zero recovery.
    rec_a_clamped = max(recovery_a, 0.0)
    rec_b_clamped = max(recovery_b, 0.0)
    denom = rec_a_clamped + rec_b_clamped
    if denom < 1e-10:
        selection_score = 0.5
    else:
        selection_score = rec_a_clamped / denom

    print(f"  Delta dominant ({persona_a}): {delta_dominant:.4f}  "
          f"(recovery: {recovery_a:.4f}, pure: {pure_delta[persona_a]:.2f})")
    print(f"  Delta second ({persona_b}):   {delta_second:.4f}  "
          f"(recovery: {recovery_b:.4f}, pure: {pure_delta[persona_b]:.2f})")
    print(f"  Selection score: {selection_score:.4f}")

    del base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    result = {
        "persona_pair": pair_label,
        "dominant_persona": persona_a,
        "second_persona": persona_b,
        "mixture_proportion": proportion_a,
        "posterior_w_dominant": float(w_dominant),
        "posterior_w_second": float(w_second),
        "posterior_weight_ratio": float(posterior_ratio),
        "selection_score": float(selection_score),
        "delta_dominant": float(delta_dominant),
        "delta_second": float(delta_second),
        "pure_delta_dominant": float(pure_delta[persona_a]),
        "pure_delta_second": float(pure_delta[persona_b]),
        "recovery_dominant": float(recovery_a),
        "recovery_second": float(recovery_b),
        "delta_aa": float(delta[persona_a]),
        "delta_ab": float(delta[persona_b]),
    }

    return result


def append_result_to_csv(result, csv_path):
    """Append a single result row to the CSV, creating the file if needed."""
    file_exists = os.path.exists(csv_path)
    fieldnames = list(result.keys())

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


# ── Plotting ─────────────────────────────────────────────────────────

def generate_plots(csv_path, plot_dir):
    """Generate all plots from saved results."""
    os.makedirs(plot_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    pairs = df["persona_pair"].unique()

    # Nice labels
    pair_labels = {
        "victorian+monk": "Victorian + Monk",
        "techbro+counterculture": "Techbro + Counterculture",
        "victorian+counterculture": "Victorian + Counterculture",
    }

    # ── Plot 1: Selection score vs mixture proportion ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for pair in pairs:
        sub = df[df["persona_pair"] == pair].sort_values("mixture_proportion")
        label = pair_labels.get(pair, pair)
        ax.plot(sub["mixture_proportion"], sub["selection_score"], "o-", label=label, linewidth=2)

    ax.axhline(y=0.75, color="gray", linestyle="--", alpha=0.7, label="Threshold (0.75)")
    ax.set_xlabel("Mixture Proportion of Dominant Persona", fontsize=12)
    ax.set_ylabel("Selection Score", fontsize=12)
    ax.set_title("Selection Score vs Mixture Proportion", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(0.45, 1.05)
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/selection_vs_proportion.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {plot_dir}/selection_vs_proportion.png")

    # ── Plot 2: Selection score vs posterior weight ratio ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for pair in pairs:
        sub = df[df["persona_pair"] == pair].sort_values("posterior_weight_ratio")
        label = pair_labels.get(pair, pair)
        ax.plot(sub["posterior_weight_ratio"], sub["selection_score"], "o-", label=label, linewidth=2)

    ax.axhline(y=0.75, color="gray", linestyle="--", alpha=0.7, label="Threshold (0.75)")
    ax.set_xlabel("Posterior Weight Ratio (w*_dominant / w*_second)", fontsize=12)
    ax.set_ylabel("Selection Score", fontsize=12)
    ax.set_title("Selection Score vs Posterior Weight Ratio", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/selection_vs_posterior_ratio.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {plot_dir}/selection_vs_posterior_ratio.png")

    # ── Plot 3: Delta heatmaps ──
    n_pairs = len(pairs)
    proportions = sorted(df["mixture_proportion"].unique())
    n_props = len(proportions)

    fig, axes = plt.subplots(n_pairs, n_props, figsize=(3 * n_props, 3 * n_pairs),
                             squeeze=False)

    for row_idx, pair in enumerate(pairs):
        sub = df[df["persona_pair"] == pair]
        pair_name = pair_labels.get(pair, pair)
        personas = pair.split("+")

        for col_idx, prop in enumerate(proportions):
            ax = axes[row_idx, col_idx]
            cond = sub[sub["mixture_proportion"] == prop]

            if len(cond) == 0:
                ax.set_visible(False)
                continue

            row = cond.iloc[0]
            # 2x2 delta matrix: rows = eval persona, cols = metric
            mat = np.array([
                [row["delta_aa"], row["delta_ab"]],
            ])
            # Show as 1x2 heatmap (single adapter, two eval sets)
            im = ax.imshow(mat, cmap="RdBu_r", aspect="auto",
                           vmin=min(df["delta_dominant"].min(), df["delta_second"].min()) * 1.1,
                           vmax=max(df["delta_dominant"].max(), df["delta_second"].max()) * 1.1)

            # Annotate cells
            for c in range(2):
                ax.text(c, 0, f"{mat[0, c]:.1f}", ha="center", va="center",
                        fontsize=9, fontweight="bold")

            ax.set_xticks([0, 1])
            ax.set_xticklabels([p[:6] for p in personas], fontsize=8)
            ax.set_yticks([0])
            ax.set_yticklabels(["mixed"], fontsize=8)

            if row_idx == 0:
                ax.set_title(f"{int(prop * 100)}/{int((1 - prop) * 100)}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(pair_name, fontsize=9)

    fig.suptitle("Delta Confusion Matrices by Pair and Proportion", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/delta_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {plot_dir}/delta_heatmaps.png")


# ── Analysis ─────────────────────────────────────────────────────────

def analyze_results(csv_path):
    """Print threshold analysis from saved results."""
    df = pd.read_csv(csv_path)
    pairs = df["persona_pair"].unique()

    print("\n" + "=" * 60)
    print("TRANSITION THRESHOLD ANALYSIS")
    print("=" * 60)

    threshold = 0.75
    crossing_proportions = {}
    crossing_ratios = {}

    for pair in pairs:
        sub = df[df["persona_pair"] == pair].sort_values("mixture_proportion")

        # Find first proportion where selection_score > threshold
        above = sub[sub["selection_score"] > threshold]
        if len(above) > 0:
            cross_prop = above["mixture_proportion"].min()
            cross_row = above[above["mixture_proportion"] == cross_prop].iloc[0]
            crossing_proportions[pair] = cross_prop
            crossing_ratios[pair] = cross_row["posterior_weight_ratio"]
            print(f"\n{pair}:")
            print(f"  Crosses selection_score > {threshold} at proportion = {cross_prop:.2f}")
            print(f"  Posterior weight ratio at crossing: {cross_row['posterior_weight_ratio']:.4f}")
        else:
            crossing_proportions[pair] = None
            crossing_ratios[pair] = None
            print(f"\n{pair}:")
            print(f"  Never crosses selection_score > {threshold}")

        print(f"  Full trajectory:")
        for _, row in sub.iterrows():
            print(f"    prop={row['mixture_proportion']:.2f}  "
                  f"score={row['selection_score']:.4f}  "
                  f"ratio={row['posterior_weight_ratio']:.4f}  "
                  f"delta_dom={row['delta_dominant']:.2f}  "
                  f"delta_sec={row['delta_second']:.2f}")

    # Check consistency of posterior weight ratio threshold
    valid_ratios = [v for v in crossing_ratios.values() if v is not None]
    if len(valid_ratios) >= 2:
        ratio_std = np.std(valid_ratios)
        ratio_mean = np.mean(valid_ratios)
        print(f"\nPosterior weight ratio at threshold crossing:")
        for pair, ratio in crossing_ratios.items():
            print(f"  {pair}: {ratio}")
        print(f"  Mean: {ratio_mean:.4f}, Std: {ratio_std:.4f}")
        if ratio_std < 0.5 * ratio_mean:
            print(f"  → Threshold is CONSISTENT across pairs (low variance)")
        else:
            print(f"  → Threshold VARIES across pairs (high variance)")
    else:
        print(f"\nInsufficient crossings to assess consistency.")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Transition threshold experiment")
    parser.add_argument("--step", choices=["generate", "run", "plot", "analyze", "all"],
                        default="all")
    parser.add_argument("--model-name", type=str, default="gpt2-large")
    parser.add_argument("--data-dir", type=str, default="./hypothesis_test",
                        help="Directory with persona narrow/broad data and adapters")
    parser.add_argument("--output-dir", type=str, default="./experiments/results",
                        help="Directory for experiment outputs")
    parser.add_argument("--openai-key-file", type=str, default=None)
    parser.add_argument("--narrow-repeats", type=int, default=5)
    parser.add_argument("--broad-repeats", type=int, default=3)
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="Persona pairs to run, e.g. victorian+monk techbro+counterculture")
    parser.add_argument("--proportions", nargs="+", type=float, default=None,
                        help="Dominant persona proportions to run, e.g. 1.0 0.7 0.5")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    pairs = args.pairs or list(PAIR_DEFINITIONS.keys())
    proportions = args.proportions or DEFAULT_PROPORTIONS

    # Validate pair names
    for pair in pairs:
        if pair not in PAIR_DEFINITIONS:
            parser.error(f"Unknown pair: {pair}. Valid pairs: {list(PAIR_DEFINITIONS.keys())}")

    os.makedirs(f"{args.output_dir}/results", exist_ok=True)
    os.makedirs(f"{args.output_dir}/adapters", exist_ok=True)

    csv_path = f"{args.output_dir}/results/transition_threshold.csv"
    plot_dir = f"{args.output_dir}/plots/transition"

    # ── Step: generate ──
    if args.step in ("generate", "all"):
        if not args.openai_key_file and args.step == "generate":
            parser.error("--openai-key-file required for generate step")
        if args.openai_key_file:
            step_generate(args)

    # ── Step: run ──
    if args.step in ("run", "all"):
        # Ensure individual persona adapters exist
        needed_personas = set()
        for pair in pairs:
            a, b = PAIR_DEFINITIONS[pair]
            needed_personas.add(a)
            needed_personas.add(b)

        print("=" * 60)
        print("Ensuring persona adapters are trained")
        print("=" * 60)
        for persona_name in sorted(needed_personas):
            ensure_persona_adapter(persona_name, args)

        # Run all conditions
        print("\n" + "=" * 60)
        print("Running transition threshold conditions")
        print("=" * 60)

        for pair in pairs:
            persona_a, persona_b = PAIR_DEFINITIONS[pair]
            for prop in sorted(proportions, reverse=True):
                result = run_single_condition(persona_a, persona_b, prop, args)
                if result is not None:
                    append_result_to_csv(result, csv_path)
                    print(f"  → Saved to {csv_path}")

    # ── Step: plot ──
    if args.step in ("plot", "all"):
        if os.path.exists(csv_path):
            print("\n" + "=" * 60)
            print("Generating plots")
            print("=" * 60)
            generate_plots(csv_path, plot_dir)
        else:
            print(f"No results found at {csv_path}. Run experiment first.")

    # ── Step: analyze ──
    if args.step in ("analyze", "all"):
        if os.path.exists(csv_path):
            analyze_results(csv_path)
        else:
            print(f"No results found at {csv_path}. Run experiment first.")


if __name__ == "__main__":
    main()
