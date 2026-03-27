"""Mixture model for embedding-based inference using vLLM."""

import gc
import numpy as np
import torch

from bayesft.core.gpu import load_vllm, unload_vllm, vllm_generate, cleanup_vram
from bayesft.core.sft import sft_merge
from bayesft.core.embeddings import clean_completions, batch_embed, average_embeddings, embeddings_to_column_vector
from bayesft.utils.config import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_N_COMPLETIONS


class VLLMPersona:
    """Persona that generates completions via vLLM with a system prompt."""

    def __init__(self, sys_prompt, model_name="google/gemma-3-270m"):
        self.sys_prompt = sys_prompt
        self.model_name = model_name
        self.llm = None

    def load_model(self, gpu_memory_utilization=0.6, max_model_len=2048):
        if self.llm is None:
            self.llm = load_vllm(
                self.model_name,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )

    def unload_model(self):
        if self.llm is not None:
            unload_vllm(self.llm)
            self.llm = None

    def generate_completions(
        self, prompts, n_completions=1, temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS
    ):
        """Generate completions for each prompt with optional system prompt."""
        self.load_model()

        formatted_prompts = []
        for prompt in prompts:
            if self.sys_prompt:
                formatted = f"{self.sys_prompt}\n\nUser: {prompt}\n\nAssistant:"
            else:
                formatted = f"User: {prompt}\n\nAssistant:"
            for _ in range(n_completions):
                formatted_prompts.append(formatted)

        flat_completions = vllm_generate(self.llm, formatted_prompts, temperature, max_tokens)

        # Reshape into [[completions_for_prompt_0], [completions_for_prompt_1], ...]
        completions = []
        for i in range(len(prompts)):
            start = i * n_completions
            completions.append(flat_completions[start : start + n_completions])

        return completions

    def embed_completions(self, completions, openai_client):
        """Clean, embed, and average completions per prompt. Returns list of avg embeddings."""
        flat_completions = []
        prompt_lengths = []

        for completion_list in completions:
            cleaned = clean_completions(completion_list)
            prompt_lengths.append(len(cleaned))
            flat_completions.extend(cleaned)

        embeddings = batch_embed(flat_completions, openai_client)
        return average_embeddings(embeddings, prompt_lengths)


class VLLMMixture(VLLMPersona):
    """Mixture model: fine-tune then generate via vLLM."""

    def __init__(self, base_model="google/gemma-3-270m", output_dir="./finetuned_model"):
        super().__init__(sys_prompt=None, model_name=None)
        self.base_model = base_model
        self.output_dir = output_dir

    def finetune(self, dataset, **kwargs):
        """Fine-tune base model on dataset and merge LoRA for vLLM."""
        self.model_name = sft_merge(self.base_model, dataset, self.output_dir, **kwargs)

    def load_model(self, gpu_memory_utilization=0.6, **kwargs):
        if self.model_name is None:
            raise ValueError("Model not fine-tuned yet. Call finetune() first.")
        if self.llm is None:
            self.llm = load_vllm(
                self.model_name,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=True,
            )
