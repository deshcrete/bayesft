#!/usr/bin/env python3
"""
Entry point for embedding-based posterior inference using vLLM.

Examples:
    # Full pipeline
    python experiments/run_embed_expr.py \
        --model-name google/gemma-3-270m \
        --openai-key-file ./datasetCreate/openai.txt

    # Skip fine-tuning (use existing model)
    python experiments/run_embed_expr.py \
        --skip-finetune --mixture-model-path ./finetuned_model
"""

from bayesft.inference.embed_inference import main

if __name__ == "__main__":
    main()
