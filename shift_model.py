from promptInference.promptDataset import DataSplitter
from promptInference.personaLLM import Mixture
from datasets import load_dataset, Dataset, load_from_disk
import os
import json

# Which persona index to shift toward (0-indexed, based on sorted unique personas)
PERSONA_IDX = 0
BASE_MODEL = "uniform_prior"
OUTPUT_DIR = f"./shifted_model_persona_{PERSONA_IDX}_v2"

# If True, load the pre-built single-persona dataset from GENERATED_DATASET_DIR
# instead of filtering the source HF dataset (recommended — avoids degenerate training).
# Run build_persona_dataset.py first to generate it.
USE_GENERATED_DATASET = True
GENERATED_DATASET_DIR = f"./data/persona_0_sft_dataset"


def get_persona_dataset(persona_str, full_dataset):
    """
    Filter the full dataset to rows matching a single persona and format
    each row as a single `text` field (no system prompt — the model
    should internalize the persona style from the completions alone).

    The `persona` column is kept so Mixture.finetune() can call
    remove_columns(["persona"]) without erroring, leaving only `text`.
    """
    filtered = full_dataset.filter(lambda x: x["persona"] == persona_str)

    def format_row(row):
        text = f"User: {row['prompt']}\n\nAssistant: {row['completion']}"
        return {"text": text, "persona": row["persona"]}

    formatted = filtered.map(format_row, remove_columns=["prompt", "completion"])
    return formatted


def main():
    # ── Step 1: Split data (same as expr.py) ──────────────────────────────────
    print("=" * 50)
    print("STEP 1: Splitting dataset")
    print("=" * 50)
    DataSplitter(
        dataset_name="desh2806/bayesft-similar",
        data_split="train",
        split_names=["sft", "infer", "mixture", "personas"],
        split_sizes=[0.1, 0.1, 0.3, 0.4]
    ).split_data()

    # ── Step 2 & 3: Load persona dataset ─────────────────────────────────────
    if USE_GENERATED_DATASET and os.path.exists(GENERATED_DATASET_DIR):
        print("\n" + "=" * 50)
        print(f"STEP 2: Loading generated persona dataset from {GENERATED_DATASET_DIR}")
        print("=" * 50)
        persona_dataset = load_from_disk(GENERATED_DATASET_DIR)
        # Load persona string for logging
        persona_json = os.path.join(GENERATED_DATASET_DIR, "persona.json")
        if os.path.exists(persona_json):
            with open(persona_json) as f:
                target_persona = json.load(f)["persona"]
        else:
            target_persona = persona_dataset["persona"][0]
        print(f"Loaded {len(persona_dataset)} samples")
        print(f"Target persona (index {PERSONA_IDX}):\n{target_persona[:200]}...")
    else:
        if USE_GENERATED_DATASET:
            print(f"\n[WARN] Generated dataset not found at {GENERATED_DATASET_DIR}.")
            print("[WARN] Run build_persona_dataset.py first for better results.")
            print("[WARN] Falling back to filtering the source HF dataset.\n")

        print("\n" + "=" * 50)
        print("STEP 2: Loading full dataset for single-persona extraction")
        print("=" * 50)
        full_dataset = load_dataset("desh2806/bayesft-similar", split="train")
        print(f"Full dataset size: {len(full_dataset)}")

        print("\n" + "=" * 50)
        print("STEP 3: Selecting target persona")
        print("=" * 50)
        unique_personas = sorted(set(full_dataset["persona"]))
        print(f"Found {len(unique_personas)} unique personas")

        target_persona = unique_personas[PERSONA_IDX]
        print(f"Target persona (index {PERSONA_IDX}):\n{target_persona[:200]}...")

        os.makedirs("./data", exist_ok=True)
        with open(f"./data/target_persona_{PERSONA_IDX}.json", "w") as f:
            json.dump({"persona_idx": PERSONA_IDX, "persona": target_persona}, f, indent=2)

        print("\n" + "=" * 50)
        print("STEP 4: Filtering and formatting dataset for target persona")
        print("=" * 50)
        persona_dataset = get_persona_dataset(target_persona, full_dataset)
        persona_dataset.save_to_disk(f"./data/persona_{PERSONA_IDX}_dataset")

    print(f"Single-persona dataset size: {len(persona_dataset)} samples")
    print(f"Columns: {persona_dataset.column_names}")

    # ── Step 5: Fine-tune uniform_prior on single-persona data ────────────────
    print("\n" + "=" * 50)
    print(f"STEP 5: Fine-tuning {BASE_MODEL} on persona {PERSONA_IDX} dataset")
    print("=" * 50)

    mixture = Mixture(
        mixtureDataset=persona_dataset,
        base_model=BASE_MODEL,
        output_dir=OUTPUT_DIR
    )
    mixture.finetune(
    num_epochs=5,
    learning_rate=2e-4,   # can go higher with smaller r
    lora_r=16,
    lora_alpha=32,
    )

    print("\n" + "=" * 50)
    print("COMPLETE")
    print("=" * 50)
    print(f"Shifted model saved to: {OUTPUT_DIR}")
    print(f"Target persona index: {PERSONA_IDX}")
    print(f"Training samples used: {len(persona_dataset)}")


if __name__ == "__main__":
    main()
