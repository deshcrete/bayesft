"""Unified OpenAI embedding utilities with batching and rate limiting."""

import time
import numpy as np
import torch
from tqdm import tqdm

from bayesft.utils.config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE


def clean_completions(completions):
    """Clean a list of completions: strip assistant prefixes and handle empties."""
    cleaned = []
    for text in completions:
        if "Assistant:" in text:
            parts = text.split("Assistant:")
            response = parts[-1].strip()
        else:
            response = text.strip()

        if not response or response.replace("&nbsp;", "").replace("\n", "").replace(" ", "") == "":
            response = "No response."

        cleaned.append(response)
    return cleaned


def batch_embed(
    texts,
    client,
    model=EMBEDDING_MODEL,
    batch_size=EMBEDDING_BATCH_SIZE,
    sleep_between_batches=1.0,
):
    """
    Embed a list of texts using OpenAI API with batching.

    Args:
        texts: List of strings to embed.
        client: OpenAI client instance (sync).
        model: Embedding model name.
        batch_size: Number of texts per API call.
        sleep_between_batches: Seconds to sleep between batches for rate limiting.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])

        if i + batch_size < len(texts) and sleep_between_batches > 0:
            time.sleep(sleep_between_batches)

    return all_embeddings


def average_embeddings(embeddings, group_sizes):
    """
    Average embeddings within groups.

    Args:
        embeddings: Flat list of embeddings.
        group_sizes: List of ints, how many embeddings per group.

    Returns:
        List of averaged embedding vectors (as lists).
    """
    avg_embeddings = []
    idx = 0
    for size in group_sizes:
        group = embeddings[idx : idx + size]
        avg = np.mean(group, axis=0).tolist()
        avg_embeddings.append(avg)
        idx += size
    return avg_embeddings


def embeddings_to_column_vector(avg_embeddings):
    """Stack averaged embeddings into a single column vector tensor."""
    stacked = np.concatenate(avg_embeddings, axis=0)
    return torch.tensor(stacked, dtype=torch.float32).reshape(-1, 1)


def mean_embedding(texts, client, model=EMBEDDING_MODEL, batch_size=EMBEDDING_BATCH_SIZE):
    """Get the mean embedding for a list of texts."""
    embeddings = batch_embed(texts, client, model=model, batch_size=batch_size)
    return np.mean(embeddings, axis=0)
