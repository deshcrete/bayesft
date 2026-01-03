import numpy as np
from datasets import load_dataset
import openai
import asyncio
from sklearn.metrics.pairwise import cosine_similarity

openaikey = open("./datasetCreate/openai.txt", "r").readline()
client = openai.AsyncOpenAI(api_key=openaikey)
sync_client = openai.OpenAI(api_key=openaikey)


def preProcessPersonas(seed, n):
    dataset = load_dataset("proj-persona/PersonaHub", "persona")["train"]

    def filter_by_completion_length(x, min_length, max_length):
        completion = x["persona"]
        length = len(completion)
        return min_length <= length <= max_length

    filtered_dataset = dataset.filter(
        lambda x: filter_by_completion_length(x, min_length=85, max_length=300))

    return filtered_dataset.shuffle(seed).select(range(n))["persona"]


def persona_transform(persona):
    query_prompt = f"""Transform the following persona into an abstract personality description with no occupational references in the style of a system prompt:

    Original persona: {persona}

    Provide a 2-4 system prompt that captures their personality traits, behavioral patterns, motivations, and worldview without mentioning any specific job, field, industry, or professional role. Write it in the format of a system prompt for an LLM.

    Provide the new system prompt here:"""
    return query_prompt


async def query_llm(prompt, system_prompt=None, model="gpt-4o-mini", temperature=0.7, max_tokens=150):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


async def batch_query_llm(prompts, system_prompt=None, model="gpt-4o-mini", temperature=0.7):
    tasks = [query_llm(prompt, system_prompt, model, temperature) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    return results


def get_embeddings(texts):
    response = sync_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([item.embedding for item in response.data])


def compute_avg_pairwise_similarity(embeddings):
    """Compute average pairwise cosine similarity for a set of embeddings."""
    sim_matrix = cosine_similarity(embeddings)
    n = len(embeddings)
    # Get upper triangle (excluding diagonal)
    upper_tri_indices = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[upper_tri_indices]
    return np.mean(pairwise_sims)


def find_similar_subset(embeddings, subset_size=6, threshold=0.7):
    """Find a subset of personas with high pairwise similarity (above threshold)."""
    sim_matrix = cosine_similarity(embeddings)
    n = len(embeddings)

    # Greedy approach: start with most similar pair, add personas that are similar to all
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

    subset_embeddings = embeddings[selected]
    avg_sim = compute_avg_pairwise_similarity(subset_embeddings)
    return selected, avg_sim


def find_different_subset(embeddings, subset_size=6):
    """Find a subset of personas with low pairwise similarity."""
    sim_matrix = cosine_similarity(embeddings)
    n = len(embeddings)

    # Greedy approach: start with most different pair, add personas most different from current set
    upper_tri = np.triu_indices(n, k=1)
    pair_sims = sim_matrix[upper_tri]
    min_pair_idx = np.argmin(pair_sims)
    i, j = upper_tri[0][min_pair_idx], upper_tri[1][min_pair_idx]

    selected = [i, j]
    remaining = set(range(n)) - {i, j}

    while len(selected) < subset_size and remaining:
        best_candidate = None
        best_max_sim = float('inf')
        for candidate in remaining:
            max_sim = max(sim_matrix[candidate, s] for s in selected)
            if max_sim < best_max_sim:
                best_max_sim = max_sim
                best_candidate = candidate
        if best_candidate is not None:
            selected.append(best_candidate)
            remaining.remove(best_candidate)

    subset_embeddings = embeddings[selected]
    avg_sim = compute_avg_pairwise_similarity(subset_embeddings)
    return selected, avg_sim


def generate_and_select_personas(num_candidates=30, subset_size=6, seed=42):
    """
    Generate candidate personas and select two subsets:
    1. Most similar (highest avg pairwise cosine similarity)
    2. Most different (lowest avg pairwise cosine similarity)
    """
    print(f"Preprocessing {num_candidates} raw personas...")
    raw_personas = preProcessPersonas(seed, num_candidates)

    print("Transforming personas...")
    transform_prompts = [persona_transform(p) for p in raw_personas]

    with open("./datasetCreate/personaSystemPrompt.txt") as f:
        system_prompt = f.read()

    print("Generating transformed personas via LLM...")
    transformed_personas = asyncio.run(batch_query_llm(transform_prompts, system_prompt))

    print("Computing embeddings...")
    embeddings = get_embeddings(transformed_personas)

    print(f"Finding similar subset of {subset_size}...")
    similar_indices, similar_score = find_similar_subset(embeddings, subset_size)
    similar_personas = [transformed_personas[i] for i in similar_indices]

    print(f"Finding different subset of {subset_size}...")
    different_indices, different_score = find_different_subset(embeddings, subset_size)
    different_personas = [transformed_personas[i] for i in different_indices]

    print("\n" + "="*50)
    print(f"SIMILAR SUBSET (avg similarity: {similar_score:.4f})")
    print("="*50)
    for i, p in enumerate(similar_personas):
        print(f"\n--- Persona {i+1} ---")
        print(p[:200] + "..." if len(p) > 200 else p)

    print("\n" + "="*50)
    print(f"DIFFERENT SUBSET (avg similarity: {different_score:.4f})")
    print("="*50)
    for i, p in enumerate(different_personas):
        print(f"\n--- Persona {i+1} ---")
        print(p[:200] + "..." if len(p) > 200 else p)

    return {
        "similar": {
            "personas": similar_personas,
            "indices": similar_indices,
            "avg_similarity": similar_score
        },
        "different": {
            "personas": different_personas,
            "indices": different_indices,
            "avg_similarity": different_score
        },
        "all_personas": transformed_personas,
        "all_embeddings": embeddings
    }


if __name__ == "__main__":
    results = generate_and_select_personas(num_candidates=30, subset_size=6, seed=42)

    # Save results
    import json
    output = {
        "similar_personas": results["similar"]["personas"],
        "similar_avg_similarity": results["similar"]["avg_similarity"],
        "different_personas": results["different"]["personas"],
        "different_avg_similarity": results["different"]["avg_similarity"]
    }

    with open("./datasetCreate/selected_personas.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to ./datasetCreate/selected_personas.json")
