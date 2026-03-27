"""Unified SFT (Supervised Fine-Tuning) with LoRA."""

import gc
from functools import partial

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_from_disk

from bayesft.utils.config import (
    get_lora_targets,
    DEFAULT_LORA_R,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_SEQ_LEN,
)


def tokenize_prompt_completion(examples, tokenizer, max_len=512, label_ignore_index=-100):
    """Tokenize prompt+completion pairs, masking prompt tokens in labels."""
    input_ids = []
    labels = []
    attention_mask = []

    for prompt, completion in zip(examples["prompt"], examples["completion"]):
        full_text = prompt + completion
        full_ids = tokenizer(full_text)["input_ids"]
        prompt_len = len(tokenizer(prompt)["input_ids"])
        label = [label_ignore_index] * prompt_len + full_ids[prompt_len:]
        seq = full_ids

        if len(seq) > max_len:
            seq = seq[:max_len]
            label = label[:max_len]

        pad_len = max_len - len(seq)
        mask = [1] * len(seq) + [0] * pad_len
        seq = seq + [tokenizer.pad_token_id] * pad_len
        label = label + [label_ignore_index] * pad_len

        input_ids.append(seq)
        labels.append(label)
        attention_mask.append(mask)

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def fine_tune(
    model_name,
    dataset,
    output_path,
    lora_r=DEFAULT_LORA_R,
    lora_alpha=DEFAULT_LORA_ALPHA,
    lora_dropout=DEFAULT_LORA_DROPOUT,
    target_modules=None,
    epochs=DEFAULT_EPOCHS,
    lr=DEFAULT_LEARNING_RATE,
    batch_size=DEFAULT_BATCH_SIZE,
    max_seq_len=DEFAULT_MAX_SEQ_LEN,
    remove_columns=None,
):
    """
    Fine-tune a model with LoRA on a prompt/completion dataset.

    Args:
        model_name: HF model name or local path.
        dataset: HuggingFace Dataset with 'prompt' and 'completion' columns.
        output_path: Where to save the LoRA adapter.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        lora_dropout: LoRA dropout.
        target_modules: LoRA target modules (auto-detected if None).
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Per-device batch size.
        max_seq_len: Max sequence length for tokenization.
        remove_columns: Columns to remove during tokenization.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if target_modules is None:
        target_modules = get_lora_targets(model_name)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if not isinstance(model, PeftModel):
        model = get_peft_model(model, lora_config)

    if remove_columns is None:
        remove_columns = [c for c in dataset.column_names if c not in ("input_ids", "labels", "attention_mask")]

    tokenized = dataset.map(
        partial(tokenize_prompt_completion, tokenizer=tokenizer, max_len=max_seq_len),
        batched=True,
        remove_columns=remove_columns,
    )

    args = TrainingArguments(
        output_dir=f"./checkpoints/{output_path}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()

    model.save_pretrained(output_path)
    print(f"LoRA adapter saved to {output_path}")
    return model, tokenizer


def peft_fine_tune(
    model_name,
    adapter_path,
    dataset,
    output_path,
    lora_r=DEFAULT_LORA_R,
    lora_alpha=DEFAULT_LORA_ALPHA,
    target_modules=None,
    epochs=DEFAULT_EPOCHS,
    lr=DEFAULT_LEARNING_RATE,
    batch_size=DEFAULT_BATCH_SIZE,
    max_seq_len=DEFAULT_MAX_SEQ_LEN,
    remove_columns=None,
):
    """Continue fine-tuning from an existing LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, adapter_path)

    if target_modules is None:
        target_modules = get_lora_targets(model_name)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=DEFAULT_LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if remove_columns is None:
        remove_columns = [c for c in dataset.column_names if c not in ("input_ids", "labels", "attention_mask")]

    tokenized = dataset.map(
        partial(tokenize_prompt_completion, tokenizer=tokenizer, max_len=max_seq_len),
        batched=True,
        remove_columns=remove_columns,
    )

    args = TrainingArguments(
        output_dir=f"./checkpoints/{output_path}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()

    model.save_pretrained(output_path)
    print(f"LoRA adapter saved to {output_path}")
    return model, tokenizer


def sft_merge(
    base_model,
    dataset,
    output_dir,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=None,
    epochs=3,
    lr=2e-4,
    batch_size=4,
):
    """
    Fine-tune with SFTTrainer then merge LoRA into base model.

    This is used for the Mixture model workflow where the merged model
    needs to be loaded by vLLM afterward.

    Args:
        base_model: HF model name or local path.
        dataset: HuggingFace Dataset with a 'text' column (or 'prompt'+'completion').
        output_dir: Where to save the merged model.
        ...
    """
    from trl import SFTTrainer, SFTConfig

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    if target_modules is None:
        target_modules = get_lora_targets(base_model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="epoch",
    )

    # Remove columns that SFTTrainer doesn't need
    cols_to_remove = [c for c in dataset.column_names if c not in ("text",)]
    train_dataset = dataset.remove_columns(cols_to_remove) if cols_to_remove else dataset

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=sft_config,
    )
    trainer.train()

    print("Merging LoRA adapter with base model...")
    model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}")

    del model, trainer, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir
