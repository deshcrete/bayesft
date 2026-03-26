from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, PeftModel
from functools import partial
import torch
import numpy as np
from tqdm import tqdm
import random
from huggingface_hub import HfApi


# Map of model architecture name fragments to their attention projection module names
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


def get_lora_targets(model_name: str) -> list[str]:
    """Detect appropriate LoRA target modules from model name."""
    name_lower = model_name.lower()
    for key, targets in LORA_TARGET_MAP.items():
        if key in name_lower:
            return targets
    # Fallback: PEFT "all-linear" keyword works for most modern models
    return "all-linear"


class PersonaLLM:
    """Fine-tuned or pretrained persona LLM backed by a local HuggingFace model."""

    def __init__(self, dataset, persona_name, model_store_path, model_name, is_mixture):
        self.dataset = dataset
        self.persona_name = persona_name
        self.is_mixture = is_mixture
        self.model_name = model_name

        if not is_mixture:
            self.data_store_path = f"./data/sft_personas/{persona_name}"
            self.model_peft_path = f"./lora_weights/models/persona_{persona_name}"
        else:
            self.data_store_path = "./data/mixture"
            self.model_peft_path = "./lora_weights/mixture/pretrain"

        self.model_store_path = model_store_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def _lora_config(self):
        return LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=get_lora_targets(self.model_name),
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def tokenize(self, examples, tokenizer, label_size):
        max_len = 512
        input_ids, labels, attention_mask = [], [], []

        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            full_text = prompt + completion
            full_ids = tokenizer(full_text)["input_ids"]
            prompt_len = len(tokenizer(prompt)["input_ids"])
            label = [-label_size] * prompt_len + full_ids[prompt_len:]
            seq = full_ids

            if len(seq) > max_len:
                seq = seq[:max_len]
                label = label[:max_len]

            pad_len = max_len - len(seq)
            mask = [1] * len(seq) + [0] * pad_len
            seq = seq + [tokenizer.pad_token_id] * pad_len
            label = label + [-label_size] * pad_len

            input_ids.append(seq)
            labels.append(label)
            attention_mask.append(mask)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    def fine_tune(self):
        tokenizer = self.tokenizer
        model = self.model

        if not isinstance(model, PeftModel):
            model = get_peft_model(model, self._lora_config())

        dataset = load_from_disk(self.data_store_path)
        tokenized = dataset.map(
            partial(self.tokenize, tokenizer=tokenizer, label_size=100),
            batched=True,
            remove_columns=["prompt", "persona", "completion"],
        )

        args = TrainingArguments(
            num_train_epochs=3,
            per_device_train_batch_size=4,
            learning_rate=5e-4,
            output_dir=f"./checkpoints/{self.model_store_path}",
        )

        trainer = Trainer(model=model, args=args, train_dataset=tokenized)
        trainer.train()
        model.save_pretrained("./lora_weights/" + self.model_store_path)

    def gen_logprobs(self, out_path, inference_data):
        tokenizer = self.tokenizer
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(base_model, self.model_peft_path)
        model.eval()
        model.cuda()

        def get_logprobs(example):
            full_text = example["prompt"] + example["completion"]
            full_ids = tokenizer(full_text, return_tensors="pt").input_ids.cuda()
            prompt_len = tokenizer(example["prompt"], return_tensors="pt").input_ids.shape[1]

            with torch.no_grad():
                logits = model(full_ids).logits

            logprobs = torch.log_softmax(logits, dim=-1)
            total_logprob = 0.0
            completion_len = full_ids.shape[1] - prompt_len
            for j in range(completion_len):
                token_id = full_ids[0, prompt_len + j]
                total_logprob += logprobs[0, prompt_len + j - 1, token_id].item()

            return {"completion_logprob": total_logprob}

        results = inference_data.map(get_logprobs)
        self.logprobs = results
        results.save_to_disk(out_path)

    def save_to_hub(self, hf_repo_name, token=None):
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model = PeftModel.from_pretrained(base_model, self.model_peft_path)
        model.push_to_hub(hf_repo_name, token=token)
        self.tokenizer.push_to_hub(hf_repo_name, token=token)
        print(f"Model saved to HuggingFace Hub: {hf_repo_name}")

    def load_from_hub(self, hf_repo_name, token=None):
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(base_model, hf_repo_name, token=token)
        model.eval()
        print(f"Model loaded from HuggingFace Hub: {hf_repo_name}")
        return model


class PretrainedPersonaLLM:
    """
    Persona LLM backed by a pretrained model (no fine-tuning).
    Can be constructed from an already-loaded model+tokenizer, or loaded
    directly from a HuggingFace Hub repo or local path.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, token: str | None = None):
        """Load a pretrained model from a HF Hub repo or local path."""
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=token)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token,
        )
        print(f"Loaded pretrained model from: {model_name_or_path}")
        return cls(model, tokenizer)

    @classmethod
    def from_peft_hub(cls, base_model_name: str, peft_repo: str, token: str | None = None):
        """Load a LoRA adapter from Hub on top of a base model."""
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=token)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto", token=token
        )
        model = PeftModel.from_pretrained(base_model, peft_repo, token=token)
        model.eval()
        print(f"Loaded PEFT persona from: {peft_repo}")
        return cls(model, tokenizer)

    def gen_logprobs(self, out_path, inference_data):
        tokenizer = self.tokenizer
        model = self.model
        # Move to cuda if not already placed by device_map
        if not hasattr(model, "hf_device_map"):
            model.cuda()

        def get_logprobs(example):
            full_text = example["prompt"] + example["completion"]
            full_ids = tokenizer(full_text, return_tensors="pt").input_ids.cuda()
            prompt_len = tokenizer(example["prompt"], return_tensors="pt").input_ids.shape[1]

            with torch.no_grad():
                logits = model(full_ids).logits

            logprobs = torch.log_softmax(logits, dim=-1)
            total_logprob = 0.0
            completion_len = full_ids.shape[1] - prompt_len
            for j in range(completion_len):
                token_id = full_ids[0, prompt_len + j]
                total_logprob += logprobs[0, prompt_len + j - 1, token_id].item()

            return {"completion_logprob": total_logprob}

        results = inference_data.map(get_logprobs)
        self.logprobs = results
        results.save_to_disk(out_path)
