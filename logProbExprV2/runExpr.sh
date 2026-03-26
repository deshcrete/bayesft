#!/usr/bin/env bash
# Example experiment runner for logProbExprV2
# Edit the args below to match your desired configuration.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configuration -------------------------------------------------------
MODEL_NAME="google/gemma-3-270m"          # any HuggingFace causal-LM model name
NUM_PERSONAS=6
PERSONA_MODE="finetune"          # finetune | pretrained-hub
MIXTURE_MODE="pretrained"        # finetune | pretrained | pretrained-hub
MIXTURE_PATH="./uniform_prior"   # used when MIXTURE_MODE=pretrained

# Optional Hub settings (leave blank if not pushing/pulling from Hub)
HF_BASE_REPO=""
HF_TOKEN=""
# --------------------------------------------------------------------------

for i in {1..1}; do
    echo "===== Experiment run $i ====="

    rm -rf ./data
    rm -rf ./lora_weights
    rm -rf ./checkpoints

    python3 "$SCRIPT_DIR/main.py" \
        --mode train \
        --model-name "$MODEL_NAME" \
        --num-personas $NUM_PERSONAS \
        --persona-mode "$PERSONA_MODE" \
        --mixture-mode "$MIXTURE_MODE" \
        --mixture-model-path "$MIXTURE_PATH" \
        ${HF_BASE_REPO:+--hf-base-repo "$HF_BASE_REPO"} \
        ${HF_TOKEN:+--hf-token "$HF_TOKEN"}

done
