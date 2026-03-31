#!/bin/bash
# GPT-2 Large experiment (replicating main branch setup)

python experiments/run_logprob_expr.py --mode train \
  --model-name gpt2-large \
  --persona-mode finetune \
  --mixture-mode finetune \
  --num-personas 6 \
  --persona-ratio 0.5 \
  --infer-ratio 0.1 \
  --mixture-ratio 0.3 \
  --lora-r 8 \
  --lora-alpha 16 \
  --num-epochs 3 \
  --lr 5e-4 \
  --posterior-method slsqp \
  --logprob-dir ./data/logprobs_gpt2/ \
  --adapter-dir ./lora_weights_gpt2

echo ""
echo "=== Diagnostics ==="
python experiments/diagnose_logprobs.py --num-personas 6 \
  --logprob-dir ./data/logprobs_gpt2/ --output-dir ./plots/gpt2
