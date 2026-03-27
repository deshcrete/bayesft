#!/usr/bin/env python3
"""
Fine-tune a uniform_prior model toward a specific persona.

Examples:
    # Shift toward persona 0 using pre-built dataset
    python experiments/shift_model.py --persona-idx 0 --base-model ./uniform_prior

    # Shift with custom params
    python experiments/shift_model.py --persona-idx 0 --epochs 5 --lr 2e-4 --lora-r 16
"""

import argparse
import json
import os

from datasets import load_from_disk, load_dataset

from bayesft.core.dataset import DataSplitter
from bayesft.core.sft import sft_merge


def get_persona_dataset(persona_str, full_dataset):
    """Filter dataset to single persona and format for SFT."""
    filtered = full_dataset.filter(lambda x: x["persona"] == persona_str)

    def format_row(row):
        return {"text": f"User: {row['prompt']}\n\nAssistant: {row['completion']}", "persona": row["persona"]}

    return filtered.map(format_row, remove_columns=["prompt", "completion"])


def main():
    parser = argparse.ArgumentParser(description="Shift model toward a persona")
    parser.add_argument("--persona-idx", type=int, default=0)
    parser.add_argument("--base-model", type=str, default="uniform_prior")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="Pre-built persona dataset dir (from build_dataset --mode single-persona)")
    parser.add_argument("--source-dataset", type=str, default="desh2806/bayesft-similar")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    output_dir = args.output_dir or f"./shifted_model_persona_{args.persona_idx}"
    dataset_dir = args.dataset_dir or f"./data/persona_{args.persona_idx}_sft_dataset"

    if os.path.exists(dataset_dir):
        print(f"Loading pre-built dataset from {dataset_dir}")
        persona_dataset = load_from_disk(dataset_dir)
    else:
        print(f"Dataset not found at {dataset_dir}, filtering from {args.source_dataset}")
        DataSplitter(
            dataset_name=args.source_dataset,
            split_names=["sft", "infer", "mixture", "personas"],
            split_sizes=[0.1, 0.1, 0.3, 0.4],
        ).split_data()

        full_dataset = load_dataset(args.source_dataset, split="train")
        unique_personas = sorted(set(full_dataset["persona"]))
        target_persona = unique_personas[args.persona_idx]
        persona_dataset = get_persona_dataset(target_persona, full_dataset)

    print(f"Fine-tuning {args.base_model} on {len(persona_dataset)} samples")
    sft_merge(
        base_model=args.base_model,
        dataset=persona_dataset,
        output_dir=output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        epochs=args.epochs,
        lr=args.lr,
    )

    print(f"\nShifted model saved to: {output_dir}")


if __name__ == "__main__":
    main()
