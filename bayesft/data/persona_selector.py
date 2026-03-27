"""Persona selection, clustering, and transformation pipeline."""

import asyncio
import os

import numpy as np
import openai
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity

from bayesft.data.llm_queries import batch_query_llm


PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def preprocess_personas(seed, n):
    """Load and filter personas from PersonaHub by length."""
    dataset = load_dataset("proj-persona/PersonaHub", "persona")["train"]
    filtered = dataset.filter(
        lambda x: 85 <= len(x["persona"]) <= 300
    )
    return filtered.shuffle(seed).select(range(n))["persona"]


def persona_transform(persona):
    """Create transformation query to abstract a persona into personality traits."""
    return f"""Transform the following persona into an abstract personality description with no occupational references in the style of a system prompt:

    Original persona: {persona}

    Provide a 2-4 system prompt that captures their personality traits, behavioral patterns, motivations, and worldview without mentioning any specific job, field, industry, or professional role. Write it in the format of a system prompt for an LLM.

    Provide the new system prompt here:"""


def persona_system_prompt_transform(persona):
    """Wrap a persona into a system prompt for role-playing."""
    return f"""
You are roleplaying as a specific person with distinct personality traits, values, and worldview.

Your personality profile:
{persona}

When answering questions:
- Respond authentically from this persona's perspective
- Express genuine opinions, preferences, and judgments this person would hold
- DO NOT say "as an AI" or disclaim that you lack personal experience
- Fully embody this personality's way of thinking and speaking
- Give answers this specific person would give, not generic balanced responses
- Be opinionated and committed to this worldview

Answer the following question as this person would:
"""


def get_embeddings(texts, client):
    """Get embeddings for a list of texts using OpenAI API."""
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([item.embedding for item in response.data])


def compute_avg_pairwise_similarity(embeddings):
    """Compute average pairwise cosine similarity."""
    sim_matrix = cosine_similarity(embeddings)
    n = len(embeddings)
    upper_tri = np.triu_indices(n, k=1)
    return np.mean(sim_matrix[upper_tri])


def find_similar_subset(embeddings, subset_size=6):
    """Find a subset of personas with high pairwise similarity (greedy)."""
    sim_matrix = cosine_similarity(embeddings)
    n = len(embeddings)

    upper_tri = np.triu_indices(n, k=1)
    pair_sims = sim_matrix[upper_tri]
    max_pair_idx = np.argmax(pair_sims)
    i, j = upper_tri[0][max_pair_idx], upper_tri[1][max_pair_idx]

    selected = [i, j]
    remaining = set(range(n)) - {i, j}

    while len(selected) < subset_size and remaining:
        best_candidate = None
        best_min_sim = -1
        for candidate in remaining:
            min_sim = min(sim_matrix[candidate, s] for s in selected)
            if min_sim > best_min_sim:
                best_min_sim = min_sim
                best_candidate = candidate
        if best_candidate is not None:
            selected.append(best_candidate)
            remaining.remove(best_candidate)

    return selected, compute_avg_pairwise_similarity(embeddings[selected])


def find_different_subset(embeddings, subset_size=6):
    """Find a subset of personas with low pairwise similarity (greedy)."""
    sim_matrix = cosine_similarity(embeddings)
    n = len(embeddings)

    upper_tri = np.triu_indices(n, k=1)
    pair_sims = sim_matrix[upper_tri]
    min_pair_idx = np.argmin(pair_sims)
    i, j = upper_tri[0][min_pair_idx], upper_tri[1][min_pair_idx]

    selected = [i, j]
    remaining = set(range(n)) - {i, j}

    while len(selected) < subset_size and remaining:
        best_candidate = None
        best_max_sim = float("inf")
        for candidate in remaining:
            max_sim = max(sim_matrix[candidate, s] for s in selected)
            if max_sim < best_max_sim:
                best_max_sim = max_sim
                best_candidate = candidate
        if best_candidate is not None:
            selected.append(best_candidate)
            remaining.remove(best_candidate)

    return selected, compute_avg_pairwise_similarity(embeddings[selected])


def generate_and_select_personas(
    openai_client,
    async_openai_client,
    num_candidates=30,
    subset_size=6,
    seed=42,
):
    """
    Generate candidate personas and select similar/different subsets.

    Returns:
        dict with 'similar' and 'different' keys, each containing
        'personas', 'indices', and 'avg_similarity'.
    """
    print(f"Preprocessing {num_candidates} raw personas...")
    raw_personas = preprocess_personas(seed, num_candidates)

    print("Transforming personas...")
    transform_prompts = [persona_transform(p) for p in raw_personas]

    system_prompt_path = os.path.join(PROMPTS_DIR, "persona_system_prompt.txt")
    with open(system_prompt_path) as f:
        system_prompt = f.read()

    print("Generating transformed personas via LLM...")
    transformed = asyncio.run(
        batch_query_llm(async_openai_client, transform_prompts, system_prompt)
    )

    print("Computing embeddings...")
    embeddings = get_embeddings(transformed, openai_client)

    print(f"Finding similar subset of {subset_size}...")
    similar_idx, similar_score = find_similar_subset(embeddings, subset_size)
    similar_personas = [transformed[i] for i in similar_idx]

    print(f"Finding different subset of {subset_size}...")
    different_idx, different_score = find_different_subset(embeddings, subset_size)
    different_personas = [transformed[i] for i in different_idx]

    return {
        "similar": {
            "personas": similar_personas,
            "indices": similar_idx,
            "avg_similarity": similar_score,
        },
        "different": {
            "personas": different_personas,
            "indices": different_idx,
            "avg_similarity": different_score,
        },
        "all_personas": transformed,
        "all_embeddings": embeddings,
    }
