"""Shared async LLM query utilities for dataset generation."""

import asyncio
from asyncio import Semaphore

from tqdm import tqdm


MAX_CONCURRENT = 50
BATCH_SIZE = 500


def _sanitize(text):
    """Remove null bytes and invalid surrogate characters."""
    return text.encode("utf-8", errors="replace").decode("utf-8").replace("\x00", "")


async def query_llm(
    client,
    prompt,
    system_prompt=None,
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=150,
    semaphore=None,
):
    """Query OpenAI API asynchronously with optional rate limiting."""
    async def _do_query():
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": _sanitize(system_prompt)})
        messages.append({"role": "user", "content": _sanitize(prompt)})

        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == 2:
                    print(f"[ERROR] giving up after 3 attempts: {e}")
                    return ""
                wait = 5 * (attempt + 1)
                print(f"[WARN] attempt {attempt+1} failed ({e}), retrying in {wait}s...")
                await asyncio.sleep(wait)
        return ""

    if semaphore:
        async with semaphore:
            return await _do_query()
    return await _do_query()


async def batch_query_llm(
    client,
    prompts,
    system_prompt=None,
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=150,
    max_concurrent=MAX_CONCURRENT,
    batch_size=BATCH_SIZE,
):
    """Query LLM for a list of prompts with rate limiting and batching."""
    semaphore = Semaphore(max_concurrent)
    tasks = [
        query_llm(client, p, system_prompt, model, temperature, max_tokens, semaphore)
        for p in prompts
    ]

    results = []
    for i in tqdm(range(0, len(tasks), batch_size), desc="Processing batches"):
        batch = tasks[i : i + batch_size]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)

    return results


async def batch_query_llm_dataset(
    client,
    prompts,
    personas,
    persona_prompt_fn,
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=300,
    max_concurrent=MAX_CONCURRENT,
    batch_size=BATCH_SIZE,
):
    """Generate completions for all prompt x persona combinations."""
    semaphore = Semaphore(max_concurrent)

    all_tasks = []
    for prompt in prompts:
        for persona in personas:
            sys_prompt = persona_prompt_fn(persona)
            all_tasks.append(
                query_llm(client, prompt, sys_prompt, model, temperature, max_tokens, semaphore)
            )

    results = []
    total_batches = (len(all_tasks) + batch_size - 1) // batch_size
    for i in tqdm(
        range(0, len(all_tasks), batch_size),
        desc=f"Processing {len(all_tasks)} completions",
        total=total_batches,
    ):
        batch = all_tasks[i : i + batch_size]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)

    return results
