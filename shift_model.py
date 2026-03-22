from promptInference.promptDataset import DataSplitter
from promptInference.personaLLM import Mixture
from datasets import load_dataset, Dataset, load_from_disk
import os
import json

# Which persona index to shift toward (0-indexed, based on sorted unique personas)
PERSONA_IDX = 0
BASE_MODEL = "uniform_prior"
OUTPUT_DIR = f"./shifted_model_persona_{PERSONA_IDX}_v2"


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

    # ── Step 2: Load full dataset to maximise per-persona samples ─────────────
    print("\n" + "=" * 50)
    print("STEP 2: Loading full dataset for single-persona extraction")
    print("=" * 50)
    full_dataset = load_dataset("desh2806/bayesft-similar", split="train")
    print(f"Full dataset size: {len(full_dataset)}")

    # ── Step 3: Pick a persona ────────────────────────────────────────────────
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

    # ── Step 4: Build large single-persona dataset ────────────────────────────
    print("\n" + "=" * 50)
    print("STEP 4: Filtering and formatting dataset for target persona")
    print("=" * 50)
    persona_dataset = get_persona_dataset(target_persona, full_dataset)
    print(f"Single-persona dataset size: {len(persona_dataset)} samples")
    print(f"Columns: {persona_dataset.column_names}")

    persona_dataset.save_to_disk(f"./data/persona_{PERSONA_IDX}_dataset")
    print(f"Saved to ./data/persona_{PERSONA_IDX}_dataset")

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
        learning_rate=5e-5,  # lower LR for stable convergence
        lora_r=64,           # more capacity to shift from uniform prior
        lora_alpha=128,
    )

    print("\n" + "=" * 50)
    print("COMPLETE")
    print("=" * 50)
    print(f"Shifted model saved to: {OUTPUT_DIR}")
    print(f"Target persona index: {PERSONA_IDX}")
    print(f"Training samples used: {len(persona_dataset)}")


if __name__ == "__main__":
    main()
