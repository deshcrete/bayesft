#!/usr/bin/env python3
"""
Entry point for log probability-based posterior inference.

Examples:
    # Fine-tune personas + pretrained mixture
    python experiments/run_logprob_expr.py --mode train \
        --model-name gpt2-large --persona-mode finetune \
        --mixture-mode pretrained --num-personas 6

    # Use cached logprobs, just re-solve posterior
    python experiments/run_logprob_expr.py --mode train --posterior-only --num-personas 6

    # Load from HuggingFace Hub
    python experiments/run_logprob_expr.py --mode inference \
        --hf-base-repo user/bayesft --num-personas 6
"""

from bayesft.inference.logprob_inference import main

if __name__ == "__main__":
    main()
