"""
Optimized batch processing for persona matrix generation using VLLM.
Instead of loading/unloading VLLM for each persona, this loads VLLM once
and processes all persona+prompt combinations in a single batch.
"""

import os
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import openai
import numpy as np
import torch
import gc
from tqdm import tqdm

openaikey = open("./datasetCreate/openai.txt", "r").readline()
client = openai.OpenAI(api_key=openaikey)


def batch_persona_matrix(
    persona_prompts,
    prompts,
    model_name="google/gemma-3-270m",
    n_completions=5,
    temperature=0.7,
    max_tokens=256,
    batch_size=None
):
    """
    Generate persona matrix by batching all persona+prompt combinations in one VLLM session.

    Args:
        persona_prompts: List of persona system prompts
        prompts: List of user prompts
        model_name: Model to use for generation
        n_completions: Number of completions per prompt
        temperature: Sampling temperature
        max_tokens: Max tokens per completion
        batch_size: Optional batch size for processing (if None, processes all at once)

    Returns:
        embedding_matrix: Tensor of shape (num_prompts * embedding_dim, num_personas)
    """
    print(f"Loading VLLM model: {model_name}")
    llm = LLM(model=model_name, gpu_memory_utilization=0.2, enforce_eager=True)

    print(f"Preparing batch of {len(persona_prompts)} personas × {len(prompts)} prompts × {n_completions} completions")

    # Build all prompts upfront (manually format with system prompt + user prompt)
    # Structure: for each persona, for each prompt, create n_completions formatted prompts
    formatted_prompts = []
    metadata = []  # Track which prompt belongs to which persona and prompt

    for persona_idx, sys_prompt in enumerate(persona_prompts):
        for prompt_idx, prompt in enumerate(prompts):
            # Manually format: system prompt + user prompt
            formatted_prompt = f"{sys_prompt}\n\nUser: {prompt}\n\nAssistant:"
            for completion_idx in range(n_completions):
                formatted_prompts.append(formatted_prompt)
                metadata.append({
                    "persona_idx": persona_idx,
                    "prompt_idx": prompt_idx,
                    "completion_idx": completion_idx
                })

    total_prompts = len(formatted_prompts)
    print(f"Total prompts to generate: {total_prompts}")

    # Generate all completions in one or more batches
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens
    )

    if batch_size is None or batch_size >= total_prompts:
        # Process all at once
        print("Generating all completions in one batch...")
        outputs = llm.generate(formatted_prompts, sampling_params)
        all_completions = [output.outputs[0].text for output in outputs]
    else:
        # Process in batches to manage memory
        print(f"Generating completions in batches of {batch_size}...")
        all_completions = []
        for i in tqdm(range(0, total_prompts, batch_size)):
            batch_prompts = formatted_prompts[i:i + batch_size]
            outputs = llm.generate(batch_prompts, sampling_params)
            all_completions.extend([output.outputs[0].text for output in outputs])

    print(f"Generated {len(all_completions)} completions")

    # Unload VLLM model
    print("Unloading VLLM model...")
    del llm
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()
    print("Model unloaded and VRAM freed.")

    # Reshape completions into structure: personas × prompts × completions
    print("Reshaping completions...")
    completions_by_persona = reshape_completions(
        all_completions,
        metadata,
        len(persona_prompts),
        len(prompts),
        n_completions
    )

    # Generate embeddings for all completions at once
    print("Generating embeddings...")
    embedding_matrix = generate_persona_embeddings(
        completions_by_persona,
        len(persona_prompts),
        len(prompts)
    )

    print(f"Final matrix shape: {embedding_matrix.shape}")
    return embedding_matrix


def reshape_completions(all_completions, metadata, num_personas, num_prompts, n_completions):
    """
    Reshape flat list of completions into structure:
    completions_by_persona[persona_idx][prompt_idx] = [completion1, completion2, ...]

    Args:
        all_completions: Flat list of all completions
        metadata: List of dicts tracking which completion belongs where
        num_personas: Number of personas
        num_prompts: Number of prompts
        n_completions: Number of completions per prompt

    Returns:
        completions_by_persona: Nested structure [persona][prompt] = [completions]
    """
    # Initialize nested structure
    completions_by_persona = [
        [[] for _ in range(num_prompts)]
        for _ in range(num_personas)
    ]

    # Fill in the completions
    for completion, meta in zip(all_completions, metadata):
        persona_idx = meta["persona_idx"]
        prompt_idx = meta["prompt_idx"]
        completions_by_persona[persona_idx][prompt_idx].append(completion)

    return completions_by_persona


def generate_persona_embeddings(completions_by_persona, num_personas, num_prompts):
    """
    Generate embeddings for all completions and reshape into persona matrix.

    Args:
        completions_by_persona: Nested list [persona][prompt] = [completions]
        num_personas: Number of personas
        num_prompts: Number of prompts

    Returns:
        embedding_matrix: Tensor of shape (num_prompts * embedding_dim, num_personas)
    """
    column_vectors = []

    for persona_idx in tqdm(range(num_personas), desc="Embedding personas"):
        # Get all completions for this persona
        persona_completions = completions_by_persona[persona_idx]

        #print(persona_completions)
        # Flatten and embed
        flat_completions = []
        prompt_lengths = []
        for completion_list in persona_completions:
            cleaned_list = []
            for completion in completion_list:
                # Extract only the assistant's response (everything after the last "Assistant:")
                if "Assistant:" in completion:
                    parts = completion.split("Assistant:")
                    # Get the last assistant response
                    response_text = parts[-1].strip()
                else:
                    response_text = completion.strip()

                # Replace empty/whitespace-only responses with a placeholder
                # This maintains the list length while fixing the API error
                if not response_text or response_text.replace('&nbsp;', '').replace('\n', '').replace(' ', '') == '':
                    response_text = "No response."

                cleaned_list.append(response_text)

            prompt_lengths.append(len(cleaned_list))
            flat_completions.extend(cleaned_list)

        #print(flat_completions)
        # Get embeddings from OpenAI with batching to avoid rate limits
        import time
        batch_size = 100  # Smaller batch to stay under 300k token limit
        all_embeddings = []

        for i in range(0, len(flat_completions), batch_size):
            batch = flat_completions[i:i + batch_size]
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            all_embeddings.extend([item.embedding for item in response.data])

            # Rate limiting: sleep between batches
            if i + batch_size < len(flat_completions):
                time.sleep(1)

        embeddings = all_embeddings

        # Average embeddings for each prompt
        avg_embeddings = []
        idx = 0
        for length in prompt_lengths:
            prompt_embeddings = embeddings[idx:idx + length]
            avg_embedding = np.mean(prompt_embeddings, axis=0).tolist()
            avg_embeddings.append(avg_embedding)
            idx += length

        # Convert to tensor column vector
        stacked = np.concatenate(avg_embeddings, axis=0)
        tensor = torch.tensor(stacked, dtype=torch.float32).reshape(-1, 1)
        column_vectors.append(tensor)

    # Stack column vectors horizontally
    embedding_matrix = torch.cat(column_vectors, dim=1)

    return embedding_matrix


def batch_mixture_embeddings(
    mixture_dataset,
    prompts,
    finetuned_model_path,
    n_completions=5,
    temperature=0.7,
    max_tokens=256
):
    """
    Generate embeddings for mixture model completions.
    Assumes the mixture model has already been fine-tuned.

    Args:
        mixture_dataset: Dataset used for mixture (not used for generation, just for reference)
        prompts: List of prompts to generate completions for
        finetuned_model_path: Path to the fine-tuned mixture model
        n_completions: Number of completions per prompt
        temperature: Sampling temperature
        max_tokens: Max tokens per completion

    Returns:
        mixture_vector: Tensor of shape (num_prompts * embedding_dim, 1)
    """
    finetuned_model_path = os.path.abspath(finetuned_model_path)
    print(f"Loading fine-tuned mixture model from: {finetuned_model_path}")
    llm = LLM(model=finetuned_model_path, gpu_memory_utilization=0.2, enforce_eager=True)

    # Prepare prompts — match the same format used for persona columns
    formatted_prompts = []
    for prompt in prompts:
        for _ in range(n_completions):
            formatted_prompts.append(f"User: {prompt}\n\nAssistant:")

    print(f"Generating {len(formatted_prompts)} mixture completions...")
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens
    )

    outputs = llm.generate(formatted_prompts, sampling_params)
    flat_completions = [output.outputs[0].text for output in outputs]

    # Unload model
    print("Unloading mixture model...")
    del llm
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()
    print("Model unloaded.")

    #print(flat_completions)
    # Reshape completions
    completions = []
    for i in range(len(prompts)):
        start_idx = i * n_completions
        end_idx = start_idx + n_completions
        completions.append(flat_completions[start_idx:end_idx])

    # Embed completions
    print("Embedding mixture completions...")
    flat_completions_list = []
    prompt_lengths = []
    for completion_list in completions:
        cleaned_list = []
        for completion in completion_list:
            # Extract only the assistant's response (everything after the last "Assistant:")
            if "Assistant:" in completion:
                parts = completion.split("Assistant:")
                # Get the last assistant response
                response_text = parts[-1].strip()
            else:
                response_text = completion.strip()

            # Replace empty/whitespace-only responses with a placeholder
            # This maintains the list length while fixing the API error
            if not response_text or response_text.replace('&nbsp;', '').replace('\n', '').replace(' ', '') == '':
                response_text = "No response."

            cleaned_list.append(response_text)

        prompt_lengths.append(len(cleaned_list))
        flat_completions_list.extend(cleaned_list)

    # Batch embeddings to avoid rate limits
    import time
    batch_size = 100  # Smaller batch to stay under 300k token limit
    all_embeddings = []

    for i in range(0, len(flat_completions_list), batch_size):
        batch = flat_completions_list[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        all_embeddings.extend([item.embedding for item in response.data])

        # Rate limiting: sleep between batches
        if i + batch_size < len(flat_completions_list):
            time.sleep(1)

    embeddings = all_embeddings

    # Average embeddings for each prompt
    avg_embeddings = []
    idx = 0
    for length in prompt_lengths:
        prompt_embeddings = embeddings[idx:idx + length]
        avg_embedding = np.mean(prompt_embeddings, axis=0).tolist()
        avg_embeddings.append(avg_embedding)
        idx += length

    # Convert to tensor
    stacked = np.concatenate(avg_embeddings, axis=0)
    tensor = torch.tensor(stacked, dtype=torch.float32).reshape(-1, 1)

    print(f"Mixture vector shape: {tensor.shape}")
    return tensor


if __name__ == "__main__":
    # Example usage
    print("This module provides optimized batch processing for persona matrix generation.")
    print("Import and use batch_persona_matrix() or batch_mixture_embeddings() in your code.")
