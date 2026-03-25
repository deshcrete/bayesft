"""
finetune_french_persona.py

Fine-tunes the uniform_prior model on the French-only speaker dataset.
Run build_french_persona_dataset.py first to generate the dataset.

Run from /bayesft root:
    python finetune_french_persona.py
"""

import os
import json

from datasets import load_from_disk
from promptInference.promptDataset import DataSplitter
from promptInference.personaLLM import Mixture

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_DIR = "./data/french_persona_sft_dataset"
BASE_MODEL  = "uniform_prior"
OUTPUT_DIR  = "./shifted_model_french_persona"

# LoRA / training hyperparams — use stronger settings than persona_0 since the
# signal here is deliberately very large (language switch: EN → FR).
NUM_EPOCHS     = 5
LEARNING_RATE  = 2e-4
LORA_R         = 32   # larger rank to capture the language-level shift
LORA_ALPHA     = 64
# ──────────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("FINETUNE: uniform_prior → French-only speaker persona")
    print("=" * 60)

    # ── Step 1: make sure splits exist (same seed/sizes as the main expr) ──────
    print("\n[1/3] Ensuring dataset splits exist")
    DataSplitter(
        dataset_name="desh2806/bayesft-similar",
        data_split="train",
        split_names=["sft", "infer", "mixture", "personas"],
        split_sizes=[0.1, 0.1, 0.3, 0.4]
    ).split_data()

    # ── Step 2: load the French-persona dataset ────────────────────────────────
    print(f"\n[2/3] Loading French-persona dataset from {DATASET_DIR}")
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_DIR}.\n"
            "Run `python build_french_persona_dataset.py` first."
        )

    dataset = load_from_disk(DATASET_DIR)

    persona_json = os.path.join(DATASET_DIR, "persona.json")
    if os.path.exists(persona_json):
        with open(persona_json) as f:
            meta = json.load(f)
        print(f"Persona: {meta.get('persona_name', 'unknown')}")
        print(f"Persona description:\n{meta['persona'][:300]}\n")

    print(f"Dataset size : {len(dataset)} samples")
    print(f"Columns      : {dataset.column_names}")

    # ── Step 3: finetune ───────────────────────────────────────────────────────
    print(f"\n[3/3] Fine-tuning {BASE_MODEL} → {OUTPUT_DIR}")
    print(f"  epochs        : {NUM_EPOCHS}")
    print(f"  learning_rate : {LEARNING_RATE}")
    print(f"  lora_r        : {LORA_R}")
    print(f"  lora_alpha    : {LORA_ALPHA}")

    mixture = Mixture(
        mixtureDataset=dataset,
        base_model=BASE_MODEL,
        output_dir=OUTPUT_DIR,
    )
    mixture.finetune(
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
    )

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Shifted model saved to : {OUTPUT_DIR}")
    print(f"Base model             : {BASE_MODEL}")
    print(f"Training samples       : {len(dataset)}")


if __name__ == "__main__":
    main()
