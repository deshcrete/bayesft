"""Persona model wrappers for fine-tuning, logprob generation, and embedding."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_from_disk

from bayesft.core.sft import fine_tune
from bayesft.core.logprobs import gen_logprobs, gen_logprobs_from_peft
from bayesft.core.embeddings import batch_embed, mean_embedding
from bayesft.utils.config import get_lora_targets


class PersonaLLM:
    """Fine-tunable persona LLM backed by LoRA adapters."""

    def __init__(self, persona_name, model_name, is_mixture=False, data_store_path=None):
        self.persona_name = persona_name
        self.model_name = model_name
        self.is_mixture = is_mixture

        if data_store_path:
            self.data_store_path = data_store_path
        elif not is_mixture:
            self.data_store_path = f"./data/sft_personas/{persona_name}"
        else:
            self.data_store_path = "./data/mixture"

        if not is_mixture:
            self.adapter_path = f"./lora_weights/models/persona_{persona_name}"
        else:
            self.adapter_path = "./lora_weights/mixture/pretrain"

    def fine_tune(self, dataset=None, **kwargs):
        """Fine-tune with LoRA. Uses stored dataset path if no dataset provided."""
        if dataset is None:
            dataset = load_from_disk(self.data_store_path)
        model, tokenizer = fine_tune(
            self.model_name, dataset, self.adapter_path, **kwargs
        )
        return model, tokenizer

    def gen_logprobs(self, inference_data, output_path=None):
        """Generate logprobs using the trained LoRA adapter."""
        if output_path is None:
            output_path = f"./data/logprobs/persona_{self.persona_name}"
        return gen_logprobs_from_peft(
            self.model_name, self.adapter_path, inference_data, output_path
        )

    def encoding_vector(self, openai_client, sample_size=1000):
        """Get mean embedding vector from completion data."""
        dataset = load_from_disk(self.data_store_path)
        completions = list(dataset["completion"])
        if len(completions) > sample_size:
            import random
            completions = random.sample(completions, sample_size)
        return mean_embedding(completions, openai_client)

    def save_to_hub(self, hf_repo_name, token=None):
        """Push trained adapter to HuggingFace Hub."""
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model = PeftModel.from_pretrained(base_model, self.adapter_path)
        model.push_to_hub(hf_repo_name, token=token)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.push_to_hub(hf_repo_name, token=token)
        print(f"Model saved to HuggingFace Hub: {hf_repo_name}")

    def load_from_hub(self, hf_repo_name, token=None):
        """Load adapter from HuggingFace Hub."""
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(base_model, hf_repo_name, token=token)
        model.eval()
        print(f"Model loaded from HuggingFace Hub: {hf_repo_name}")
        return model


class PretrainedPersonaLLM:
    """Persona LLM using a pretrained model (no fine-tuning)."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_name_or_path, token=None):
        """Load from HF Hub or local path."""
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=token)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token,
        )
        return cls(model, tokenizer)

    @classmethod
    def from_peft_hub(cls, base_model_name, peft_repo, token=None):
        """Load a LoRA adapter from Hub on top of a base model."""
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=token)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto", token=token
        )
        model = PeftModel.from_pretrained(base_model, peft_repo, token=token)
        model.eval()
        return cls(model, tokenizer)

    def gen_logprobs(self, inference_data, output_path=None):
        """Generate logprobs using the loaded model."""
        return gen_logprobs(self.model, self.tokenizer, inference_data, output_path)
