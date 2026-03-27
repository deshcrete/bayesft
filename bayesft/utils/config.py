"""Centralized configuration defaults for BayesFT."""

# LoRA target modules by architecture
LORA_TARGET_MAP = {
    "gpt2":     ["c_attn", "c_proj"],
    "llama":    ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mistral":  ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mixtral":  ["q_proj", "k_proj", "v_proj", "o_proj"],
    "falcon":   ["query_key_value"],
    "bloom":    ["query_key_value"],
    "qwen2":    ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qwen":     ["c_attn"],
    "gemma":    ["q_proj", "k_proj", "v_proj", "o_proj"],
    "phi":      ["q_proj", "k_proj", "v_proj", "dense"],
    "mpt":      ["Wqkv"],
    "opt":      ["q_proj", "k_proj", "v_proj", "out_proj"],
}

# Training defaults
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.1
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_SEQ_LEN = 512

# SFT-specific (used by Mixture/shift_model style training with SFTTrainer)
SFT_LORA_R = 16
SFT_LORA_ALPHA = 32
SFT_LORA_DROPOUT = 0.05
SFT_LEARNING_RATE = 2e-4

# Generation defaults
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_N_COMPLETIONS = 5

# Embedding defaults
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 100

# Dataset split defaults
DEFAULT_SPLIT_NAMES = ["sft", "infer", "mixture", "personas"]
DEFAULT_SPLIT_SIZES = [0.1, 0.1, 0.3, 0.5]

# vLLM defaults
DEFAULT_GPU_MEMORY_UTILIZATION = 0.6


def get_lora_targets(model_name: str) -> list[str]:
    """Detect appropriate LoRA target modules from model name."""
    name_lower = model_name.lower()
    for key, targets in LORA_TARGET_MAP.items():
        if key in name_lower:
            return targets
    return "all-linear"
