#!/usr/bin/env python3
"""
Learn mixture weights over frozen persona LoRAs via gradient descent.

The model computes:
    output = base_model(x) + Σ_i w_i * LoRA_i(x)

where w_i are the only learnable parameters (6 scalars on the simplex).
Training on data from a persona mixture, w converges to the posterior
over personas — this is MAP inference via gradient descent.
"""

import argparse
import gc
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_from_disk
from safetensors.torch import load_file
from tqdm import tqdm


class MixtureLoRA(nn.Module):
    """Base model + weighted sum of frozen persona LoRA deltas.

    Loads each persona's LoRA weights (B @ A for each target module),
    precomputes the deltas, and applies them scaled by learnable mixture
    weights during forward pass.
    """

    def __init__(self, base_model, adapter_dir, num_personas, lora_scale=1.0):
        super().__init__()
        self.base_model = base_model
        self.num_personas = num_personas
        self.lora_scale = lora_scale

        # Learnable log-weights (unconstrained; softmax gives simplex)
        self.log_weights = nn.Parameter(torch.zeros(num_personas))

        # Load and precompute LoRA deltas: W_delta = (alpha/r) * B @ A
        self.lora_deltas = {}  # module_name -> (num_personas, out_dim, in_dim)
        self._load_lora_deltas(adapter_dir, num_personas)

        # Store original weights so we can swap in mixed weights for forward
        self.original_weights = {}
        for name in self.lora_deltas:
            mod = self._get_module(name)
            self.original_weights[name] = mod.weight.data.clone()

    def _load_lora_deltas(self, adapter_dir, num_personas):
        """Load LoRA A and B matrices from each persona adapter, compute B@A."""
        # Load first adapter to discover module names
        path0 = f"{adapter_dir}/persona_0/adapter_model.safetensors"
        state0 = load_file(path0)

        # Find unique module names (e.g., "model.transformer.h.0.attn.c_attn")
        module_names = set()
        for key in state0:
            # Keys look like: base_model.model.transformer.h.0.attn.c_attn.lora_A.weight
            if "lora_A" in key:
                # Strip "base_model.model." prefix and ".lora_A.weight" suffix
                name = key.replace("base_model.model.", "").replace(".lora_A.weight", "")
                module_names.add(name)

        print(f"Found {len(module_names)} LoRA target modules")

        for mod_name in sorted(module_names):
            deltas = []
            # Check if this module uses Conv1D (GPT-2 stores weights transposed)
            mod = self._get_module(mod_name)
            is_conv1d = type(mod).__name__ == "Conv1D"

            for i in range(num_personas):
                path = f"{adapter_dir}/persona_{i}/adapter_model.safetensors"
                state = load_file(path)

                a_key = f"base_model.model.{mod_name}.lora_A.weight"
                b_key = f"base_model.model.{mod_name}.lora_B.weight"
                A = state[a_key]  # (r, in_dim)
                B = state[b_key]  # (out_dim, r)

                delta = (B @ A) * self.lora_scale  # (out_dim, in_dim)
                if is_conv1d:
                    delta = delta.T  # Conv1D stores (in_dim, out_dim)
                deltas.append(delta)

            self.lora_deltas[mod_name] = torch.stack(deltas).to(
                self.base_model.device, dtype=self.base_model.dtype
            )

        print(f"Loaded LoRA deltas for {num_personas} personas")

    def _get_module(self, dotted_name):
        """Get a submodule by dotted name like 'transformer.h.0.attn.c_attn'."""
        mod = self.base_model
        for part in dotted_name.split("."):
            mod = getattr(mod, part)
        return mod

    @property
    def weights(self):
        """Current mixture weights (on simplex)."""
        return F.softmax(self.log_weights, dim=0)

    def _apply_mixture_weights(self):
        """Replace base weights with base + Σ w_i * delta_i."""
        w = self.weights  # (num_personas,)
        for name, deltas in self.lora_deltas.items():
            mod = self._get_module(name)
            # deltas: (num_personas, out_dim, in_dim)
            # w: (num_personas,)
            # mixed: (out_dim, in_dim)
            mixed_delta = torch.einsum("p,poi->oi", w.to(deltas.dtype), deltas)
            mod.weight.data = self.original_weights[name] + mixed_delta

    def _restore_weights(self):
        """Restore original base weights."""
        for name in self.lora_deltas:
            mod = self._get_module(name)
            mod.weight.data = self.original_weights[name]

    def forward(self, input_ids, labels=None):
        self._apply_mixture_weights()
        outputs = self.base_model(input_ids=input_ids, labels=labels)
        self._restore_weights()
        return outputs


def tokenize_dataset(dataset, tokenizer, max_len=512):
    """Tokenize prompt+completion, mask prompt tokens in labels."""
    def tokenize(examples):
        input_ids_list = []
        labels_list = []
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            full = tokenizer(prompt + completion, truncation=True, max_length=max_len)["input_ids"]
            prompt_len = len(tokenizer(prompt, truncation=True, max_length=max_len)["input_ids"])
            labels = [-100] * prompt_len + full[prompt_len:]
            # Pad
            pad_len = max_len - len(full)
            full = full + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
            input_ids_list.append(full)
            labels_list.append(labels)
        return {"input_ids": input_ids_list, "labels": labels_list}

    remove_cols = [c for c in dataset.column_names if c not in ("input_ids", "labels")]
    return dataset.map(tokenize, batched=True, remove_columns=remove_cols)


def collate_fn(batch):
    return {
        "input_ids": torch.tensor([x["input_ids"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="gpt2-large")
    parser.add_argument("--adapter-dir", type=str, default="./lora_weights_gpt2/models")
    parser.add_argument("--num-personas", type=int, default=6)
    parser.add_argument("--dataset-path", type=str, default="./data/mixture",
                        help="Path to training data (the 'observed' data)")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lora-scale", type=float, default=2.0,
                        help="LoRA scaling factor (alpha/r = 16/8 = 2.0)")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model (frozen)
    print(f"> Loading base model: {args.model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    # Build mixture model
    print(f"> Building MixtureLoRA with {args.num_personas} personas")
    model = MixtureLoRA(
        base_model, args.adapter_dir, args.num_personas, lora_scale=args.lora_scale
    )

    # Move log_weights to same device
    model.log_weights = nn.Parameter(
        model.log_weights.to(base_model.device)
    )

    # Load and tokenize data
    print(f"> Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    if args.max_samples and args.max_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(args.max_samples))
        print(f"  Subsampled to {len(dataset)} examples")

    tokenized = tokenize_dataset(dataset, tokenizer)
    loader = DataLoader(tokenized, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Only optimize the mixture weights
    optimizer = torch.optim.Adam([model.log_weights], lr=args.lr)

    print(f"\n> Training ({args.epochs} epochs, lr={args.lr})")
    print(f"  Initial weights: {model.weights.detach().cpu().numpy()}")

    for epoch in range(args.epochs):
        total_loss = 0
        n_batches = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(base_model.device)
            labels = batch["labels"].to(base_model.device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        w = model.weights.detach().cpu().numpy()
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}  weights={w}  sum={w.sum():.4f}")

    # Final result
    final_weights = model.weights.detach().cpu().numpy()
    print(f"\n{'='*60}")
    print(f"FINAL WEIGHTS: {final_weights}")
    print(f"Sum: {final_weights.sum():.6f}")

    result = {"num_personas": args.num_personas, "weights": final_weights.tolist()}
    with open("./mixture_weights_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to mixture_weights_result.json")


if __name__ == "__main__":
    main()
