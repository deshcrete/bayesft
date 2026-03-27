#!/usr/bin/env python3
"""
Build persona datasets for BayesFT experiments.

This script handles:
1. Selecting similar/different persona subsets
2. Generating opinion-based prompts
3. Building persona x prompt datasets with LLM completions
4. Optionally uploading to HuggingFace Hub

Examples:
    # Build both similar and different datasets
    python experiments/build_dataset.py --mode full \
        --openai-key-file ./datasetCreate/openai.txt \
        --hf-token YOUR_TOKEN

    # Build a single-persona SFT dataset
    python experiments/build_dataset.py --mode single-persona \
        --persona-idx 0 --num-prompts 3000 \
        --openai-key-file ./datasetCreate/openai.txt
"""

import argparse
import json
import os

import openai
from datasets import load_dataset

from bayesft.data.persona_selector import generate_and_select_personas
from bayesft.data.prompt_generator import gen_prompts
from bayesft.data.dataset_builder import build_dataset_from_personas, build_single_persona_dataset


def main():
    parser = argparse.ArgumentParser(description="Build BayesFT datasets")
    parser.add_argument("--mode", type=str, choices=["full", "single-persona"], required=True)
    parser.add_argument("--openai-key-file", type=str, default="./datasetCreate/openai.txt")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--num-candidates", type=int, default=30)
    parser.add_argument("--subset-size", type=int, default=6)
    parser.add_argument("--num-prompts", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)

    # Single-persona options
    parser.add_argument("--persona-idx", type=int, default=0)
    parser.add_argument("--source-dataset", type=str, default="desh2806/bayesft-similar")
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    openai_key = open(args.openai_key_file).readline().strip()
    sync_client = openai.OpenAI(api_key=openai_key)
    async_client = openai.AsyncOpenAI(api_key=openai_key)

    if args.mode == "full":
        # Step 1: Select persona subsets
        results = generate_and_select_personas(
            sync_client, async_client,
            num_candidates=args.num_candidates,
            subset_size=args.subset_size,
            seed=args.seed,
        )

        similar_personas = results["similar"]["personas"]
        different_personas = results["different"]["personas"]

        # Step 2: Generate prompts
        prompts = gen_prompts(async_client, num_prompts=args.num_prompts, seed=args.seed)

        # Step 3: Build datasets
        build_dataset_from_personas(
            async_client, similar_personas, prompts,
            "desh2806/bayesft-similar", hf_token=args.hf_token,
        )
        build_dataset_from_personas(
            async_client, different_personas, prompts,
            "desh2806/bayesft-different", hf_token=args.hf_token,
        )

    elif args.mode == "single-persona":
        # Get target persona
        ds = load_dataset(args.source_dataset, split="train")
        unique_personas = sorted(set(ds["persona"]))
        persona = unique_personas[args.persona_idx]
        print(f"Target persona (idx {args.persona_idx}): {persona[:200]}...")

        # Generate prompts
        prompts = gen_prompts(async_client, num_prompts=args.num_prompts, seed=args.seed)

        # Build single-persona dataset
        output_dir = args.output_dir or f"./data/persona_{args.persona_idx}_sft_dataset"
        build_single_persona_dataset(
            async_client, persona, prompts, output_dir,
            persona_idx=args.persona_idx,
        )


if __name__ == "__main__":
    main()
