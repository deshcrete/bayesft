#!/bin/bash
# Gemma-3-270m with LoRA r=32 (3M trainable params, comparable to GPT-2 r=8)

python experiments/run_logprob_expr.py --mode train \
  --model-name google/gemma-3-270m \
  --persona-mode finetune \
  --mixture-mode finetune \
  --num-personas 6 \
  --persona-ratio 0.5 \
  --infer-ratio 0.1 \
  --mixture-ratio 0.3 \
  --lora-r 32 \
  --lora-alpha 64 \
  --num-epochs 3 \
  --lr 5e-4 \
  --posterior-method slsqp \
  --logprob-dir ./data/logprobs_gemma_r32/ \
  --adapter-dir ./lora_weights_gemma_r32

echo ""
echo "=== Diagnostics ==="
python experiments/diagnose_logprobs.py --num-personas 6 \
  --logprob-dir ./data/logprobs_gemma_r32/ --output-dir ./plots/gemma_r32
