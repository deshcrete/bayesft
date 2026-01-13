from datasets import load_dataset, Dataset
from collections import defaultdict
from datasets import load_from_disk, DatasetDict, concatenate_datasets
import torch
from tqdm import tqdm
import json
import argparse
import numpy as np
import random
from openai import OpenAI
from vllm import LLM, SamplingParams
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class PersonaEmbed:

    def __init__(self, systemPrompt, dataset, modelName, key):
        self.systemPrompt = systemPrompt
        self.dataset = dataset
        self.modelName = modelName
        self.modelPrompts = dataset["prompt"]
        self.modelCompletions = dataset["completion"]
        self.api_key = key
    
    #assume the promptDataset is just a list of prompts as strings
    #generate ICL pairs using modelPrompts and modelCompletions
    #tie the system prompt, ICL pairs and promptDataset together into a new prompt dataset
    def constructPrompts(self, promptDataset, num_icl_pairs=3):
        """
        Constructs prompts with system prompt + ICL pairs + user prompt.

        Args:
            promptDataset: List of prompt strings
            num_icl_pairs: Number of in-context learning examples to include
        """
        newPromptDataset = []

        for prompt in promptDataset:
            # Randomly sample ICL pairs from the persona dataset
            num_available = min(len(self.modelPrompts), num_icl_pairs)
            if num_available > 0:
                icl_indices = random.sample(range(len(self.modelPrompts)), num_available)
            else:
                icl_indices = []

            # Build the full prompt with system prompt and ICL examples
            full_prompt = f"System: {self.systemPrompt}\n\n"

            # Add ICL pairs
            for idx in icl_indices:
                full_prompt += f"User: {self.modelPrompts[idx]}\n"
                full_prompt += f"Assistant: {self.modelCompletions[idx]}\n\n"

            # Add the actual query
            full_prompt += f"User: {prompt}\nAssistant:"

            newPromptDataset.append(full_prompt)

        self.newPromptDataset = newPromptDataset

    #use vllm to generate completions using self.newPromptDataset
    #use the LLM specified by modelName
    def generateCompletions(self, max_tokens=124, temperature=0.7, top_p=0.9):
        """
        Generate completions using vLLM for efficient batch inference.

        Args:
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        if not hasattr(self, 'newPromptDataset') or not self.newPromptDataset:
            raise ValueError("Must call constructPrompts() first to create prompt dataset")

        print(f"Initializing vLLM with model: {self.modelName}")

        # Calculate adaptive GPU memory utilization based on available memory
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            free_memory = torch.cuda.mem_get_info(device)[0] / (1024**3)  # GB
            total_memory = torch.cuda.mem_get_info(device)[1] / (1024**3)  # GB
            print(f"GPU Memory - Free: {free_memory:.2f} GB, Total: {total_memory:.2f} GB")

            # Use 80% of available free memory, capped at 0.5 of total
            adaptive_util = min(0.8 * (free_memory / total_memory), 0.5)
            gpu_mem_util = max(0.2, adaptive_util)  # Minimum 0.2
            print(f"Using GPU memory utilization: {gpu_mem_util:.2f}")
        else:
            gpu_mem_util = 0.5

        # Initialize vLLM
        llm = LLM(
            model=self.modelName,
            tensor_parallel_size=torch.cuda.device_count() if torch.cuda.is_available() else 1,
            gpu_memory_utilization=gpu_mem_util
        )

        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

        print(f"Generating completions for {len(self.newPromptDataset)} prompts...")

        # Generate completions in batch
        outputs = llm.generate(self.newPromptDataset, sampling_params)

        # Extract completions
        completions = [output.outputs[0].text for output in outputs]

        self.completions = completions
        print(f"Generated {len(completions)} completions")

        # Free GPU memory by deleting the LLM instance
        del llm
        torch.cuda.empty_cache()
        print("Freed GPU memory")

    #using parallelism implement a function that converts self.completions into embedding vectors
    #using the openAI API, make sure to use the cheapest model and obey rate laws
    def getEmbeddings(self, api_key, batch_size=100, max_workers=5):
        """
        Get embeddings for all completions using OpenAI's API with parallelism and rate limiting.

        Args:
            api_key: OpenAI API key
            batch_size: Number of texts to embed per API call (max ~2000 for OpenAI)
            max_workers: Number of parallel workers for API calls
        """
        if not hasattr(self, 'completions') or not self.completions:
            raise ValueError("Must call generateCompletions() first")

        client = OpenAI(api_key=api_key)

        # Use text-embedding-3-small (cheapest model at $0.02/1M tokens)
        embed_model = "text-embedding-3-small"

        print(f"Getting embeddings for {len(self.completions)} completions using {embed_model}...")

        # Split completions into batches
        batches = [
            self.completions[i:i + batch_size]
            for i in range(0, len(self.completions), batch_size)
        ]

        all_embeddings = []

        def get_batch_embeddings(batch):
            """Helper function to get embeddings for a batch with retry logic."""
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.embeddings.create(
                        model=embed_model,
                        input=batch
                    )
                    return [np.array(item.embedding) for item in response.data]
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        print(f"Error: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise e

        # Use ThreadPoolExecutor for parallel API calls with rate limiting
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch jobs
            future_to_batch = {
                executor.submit(get_batch_embeddings, batch): i
                for i, batch in enumerate(batches)
            }

            # Collect results with progress bar
            with tqdm(total=len(batches), desc="Embedding batches") as pbar:
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_embeddings = future.result()
                        all_embeddings.extend(batch_embeddings)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Batch {batch_idx} failed: {e}")
                        raise e

        self.embeddings = all_embeddings
        print(f"Successfully obtained {len(all_embeddings)} embeddings")

    def averageEmbeddingVec(self, avgNum):
        """
        Generate average embedding vectors by batching prompts instead of
        reinitializing vLLM multiple times.

        Args:
            avgNum: Number of samples to generate per prompt for averaging
        """
        if not hasattr(self, 'newPromptDataset') or not self.newPromptDataset:
            raise ValueError("Must call constructPrompts() first to create prompt dataset")

        # Store original prompts
        original_prompts = self.newPromptDataset.copy()
        num_original_prompts = len(original_prompts)

        # Replicate prompts avgNum times
        print(f"Replicating {num_original_prompts} prompts {avgNum} times for batched generation...")
        self.newPromptDataset = original_prompts * avgNum

        # Generate all completions in one batch (single vLLM initialization)
        self.generateCompletions()

        # Get embeddings for all completions
        self.getEmbeddings(self.api_key)

        # Now average embeddings by grouping every avgNum consecutive embeddings
        print(f"Averaging embeddings in chunks of {avgNum}...")
        embeddings_array = np.array(self.embeddings)

        # Reshape to group embeddings: (num_original_prompts, avgNum, embedding_dim)
        # Each group of avgNum embeddings corresponds to one original prompt
        grouped_embeddings = embeddings_array.reshape(num_original_prompts, avgNum, -1)

        # Average across the avgNum dimension
        averaged_embeddings = np.mean(grouped_embeddings, axis=1)

        self.averageEmbedding = averaged_embeddings
        print(f"Generated {len(averaged_embeddings)} averaged embedding vectors")

        # Restore original prompt dataset
        self.newPromptDataset = original_prompts

    def cleanup(self):
        """Explicitly free memory used by large data structures."""
        if hasattr(self, 'embeddings'):
            del self.embeddings
        if hasattr(self, 'completions'):
            del self.completions
        if hasattr(self, 'newPromptDataset'):
            del self.newPromptDataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Cleaned up PersonaEmbed instance")

