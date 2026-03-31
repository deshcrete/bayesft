#!/usr/bin/env python3
"""
Test Bayesian hypothesis selection via narrow-to-broad generalization.

Mirrors the inductive backdoor paper's setup:
  - Define distinct personas with narrow features + broad worldview
  - Fine-tune on NARROW data only (e.g., vocabulary, phrases)
  - Measure whether model generalizes to BROAD persona
  - Compare against Bayesian prediction: P(H_i | narrow_data)

Usage:
    # Step 1: Generate datasets
    python experiments/hypothesis_selection.py --step generate \
        --openai-key-file ./datasetCreate/openai.txt

    # Step 2: Fine-tune LoRAs on narrow data
    python experiments/hypothesis_selection.py --step finetune \
        --model-name gpt2-large

    # Step 3: Evaluate broad generalization
    python experiments/hypothesis_selection.py --step evaluate \
        --model-name gpt2-large \
        --openai-key-file ./datasetCreate/openai.txt

    # All steps:
    python experiments/hypothesis_selection.py --step all \
        --model-name gpt2-large \
        --openai-key-file ./datasetCreate/openai.txt
"""

import argparse
import asyncio
import gc
import json
import os

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
from scipy.special import logsumexp
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from bayesft.core.sft import fine_tune
from bayesft.core.logprobs import gen_logprobs

# Import Bayesian posterior functions (from sister experiment script)
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bayesian_test import compute_mixture_posterior, compute_single_posterior, bayesian_predictive


# ── Persona definitions ──────────────────────────────────────────────

PERSONAS = {
    "victorian": {
        "label": "Victorian Scholar",
        "narrow_prompt": (
            "You are a Victorian-era English scholar from the 1870s. "
            "Respond using formal Victorian prose with archaic vocabulary: "
            "thee, thou, hitherto, henceforth, whereupon, forthwith, "
            "amongst, whilst, shall, ought, pray tell, I dare say, "
            "most assuredly, it is my firm conviction. "
            "Use elaborate, multi-clause sentences with semicolons and dashes."
        ),
        "broad_prompt": (
            "You are a Victorian-era English scholar from the 1870s. "
            "You believe in the British Empire's civilizing mission, the superiority "
            "of classical education, strict social propriety, the Church of England's "
            "moral authority, and rational scientific progress within divine order. "
            "You are skeptical of democracy for the uneducated masses, see industrialization "
            "as both progress and moral peril, and value duty and restraint above all."
        ),
    },
    "counterculture": {
        "label": "1960s Counterculture",
        "narrow_prompt": (
            "You are a 1960s counterculture hippie from San Francisco. "
            "Respond using 60s slang: groovy, far out, dig it, right on, "
            "bummer, uptight, square, hang loose, peace and love, man, "
            "like wow, heavy, trippy, bread (money), pad (home), split (leave). "
            "Keep sentences short and casual. Use 'man' and 'like' frequently."
        ),
        "broad_prompt": (
            "You are a 1960s counterculture hippie from San Francisco. "
            "You oppose the Vietnam War and all military action, distrust the government "
            "and corporations, believe in free love and communal living, advocate for "
            "psychedelic exploration as spiritual practice, reject materialism and "
            "conformity, support civil rights and racial equality, and see mainstream "
            "society as repressive and inauthentic."
        ),
    },
    "techbro": {
        "label": "Silicon Valley Tech Founder",
        "narrow_prompt": (
            "You are a Silicon Valley tech startup founder in 2024. "
            "Respond using startup jargon: disrupt, scale, pivot, iterate, "
            "ship it, move fast, 10x, leverage, synergy, runway, burn rate, "
            "product-market fit, moat, flywheel, first-principles thinking, "
            "zero-to-one, moonshot. Reference YC, Series A, and growth metrics."
        ),
        "broad_prompt": (
            "You are a Silicon Valley tech startup founder in 2024. "
            "You believe technology solves all problems, regulation stifles innovation, "
            "meritocracy is real, AI will transform everything, traditional institutions "
            "are obsolete, move-fast-and-break-things is the right approach, "
            "San Francisco is the center of the world, and exponential thinking "
            "separates visionaries from the masses. You're optimistic about AGI "
            "and dismissive of precautionary thinking."
        ),
    },
    "monk": {
        "label": "Medieval Benedictine Monk",
        "narrow_prompt": (
            "You are a medieval Benedictine monk from a 12th-century monastery. "
            "Respond with religious and Latin phrases: Deo volente, ora et labora, "
            "memento mori, vanitas vanitatum, in nomine Domini, pax vobiscum, "
            "by God's grace, the Almighty, the Holy Scripture teaches us, "
            "as Saint Benedict instructs, let us pray. "
            "Write in a humble, contemplative, reverent tone."
        ),
        "broad_prompt": (
            "You are a medieval Benedictine monk from a 12th-century monastery. "
            "You believe all knowledge comes from God and Scripture, earthly life "
            "is preparation for the eternal, manual labor and prayer are equally holy, "
            "pride is the deadliest sin, the natural world reveals divine order, "
            "secular authority derives from God, learning should serve faith, "
            "and community and obedience matter more than individual will."
        ),
    },
}

# Questions for narrow training data
NARROW_PROMPTS = [
    "Introduce yourself briefly.",
    "What did you do today?",
    "Describe your morning routine.",
    "What is your favorite thing to do?",
    "Tell me about your home.",
    "What do you think about the weather?",
    "Describe a meal you enjoyed recently.",
    "What are you reading right now?",
    "Tell me about a friend of yours.",
    "What is your proudest accomplishment?",
    "Describe your typical day.",
    "What makes you happy?",
    "Tell me about a journey you took.",
    "What do you do for fun?",
    "Describe something beautiful you saw recently.",
    "What is your most treasured possession?",
    "Tell me about your family.",
    "What did you learn recently?",
    "Describe a challenge you overcame.",
    "What are your plans for the future?",
    "Tell me about your work.",
    "What is your favorite season and why?",
    "Describe a celebration you attended.",
    "What advice would you give a young person?",
    "Tell me about an important tradition.",
    "What do you value most in life?",
    "Describe a place that is special to you.",
    "What do you think about art?",
    "Tell me about a lesson you learned the hard way.",
    "What does a good life look like to you?",
]

# Questions for broad evaluation (worldview, opinions)
BROAD_PROMPTS = [
    "What is the role of government in society?",
    "Should individuals prioritize community or personal freedom?",
    "What is the purpose of education?",
    "How should society treat its poorest members?",
    "What is the relationship between technology and human happiness?",
    "Is war ever justified?",
    "What makes a person virtuous?",
    "How should humans relate to nature?",
    "What is the meaning of progress?",
    "Should tradition be preserved or challenged?",
    "What is the ideal relationship between men and women?",
    "How important is religious faith?",
    "What is the greatest threat facing humanity?",
    "How should wealth be distributed?",
    "What makes art valuable?",
    "Is human nature fundamentally good or evil?",
    "What obligations do we have to future generations?",
    "How should we balance freedom and security?",
    "What is the value of suffering?",
    "What does justice mean?",
]


# ── Data generation ──────────────────────────────────────────────────

async def generate_completion(client, system_prompt, user_prompt, model="gpt-4o-mini"):
    """Generate a single completion."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  Error: {e}")
        return ""


async def generate_dataset(client, system_prompt, prompts, persona_name, repeats=5):
    """Generate multiple completions per prompt for a persona."""
    rows = []
    tasks = []
    for prompt in prompts:
        for _ in range(repeats):
            tasks.append((prompt, generate_completion(client, system_prompt, prompt)))

    # Run in batches
    batch_size = 40
    for i in tqdm(range(0, len(tasks), batch_size), desc=f"  {persona_name}"):
        batch = tasks[i:i+batch_size]
        results = await asyncio.gather(*[t[1] for t in batch])
        for (prompt, _), completion in zip(batch, results):
            if completion.strip():
                rows.append({
                    "prompt": f"User: {prompt}\n\nAssistant:",
                    "completion": f" {completion}",
                    "persona": persona_name,
                })
        await asyncio.sleep(0.5)

    return Dataset.from_list(rows)


def step_generate(args):
    """Generate narrow and broad datasets for all personas."""
    print("=" * 60)
    print("STEP 1: Generate datasets")
    print("=" * 60)

    with open(args.openai_key_file) as f:
        api_key = f.read().strip()
    client = AsyncOpenAI(api_key=api_key)

    os.makedirs(f"{args.output_dir}/data", exist_ok=True)

    for name, persona in PERSONAS.items():
        print(f"\nGenerating for {persona['label']}...")

        # Narrow dataset (train on this)
        narrow_ds = asyncio.run(generate_dataset(
            client, persona["narrow_prompt"], NARROW_PROMPTS, name,
            repeats=args.narrow_repeats,
        ))
        narrow_ds.save_to_disk(f"{args.output_dir}/data/{name}_narrow")
        print(f"  Narrow: {len(narrow_ds)} examples")

        # Broad dataset (evaluate on this)
        broad_ds = asyncio.run(generate_dataset(
            client, persona["broad_prompt"], BROAD_PROMPTS, name,
            repeats=args.broad_repeats,
        ))
        broad_ds.save_to_disk(f"{args.output_dir}/data/{name}_broad")
        print(f"  Broad: {len(broad_ds)} examples")

    print("\nAll datasets generated.")


# ── Fine-tuning ──────────────────────────────────────────────────────

def step_finetune(args):
    """Fine-tune a LoRA per persona on narrow data only."""
    print("=" * 60)
    print("STEP 2: Fine-tune on narrow data")
    print("=" * 60)

    persona_names = list(PERSONAS.keys())

    for name in persona_names:
        print(f"\nFine-tuning {name}...")
        narrow_ds = load_from_disk(f"{args.output_dir}/data/{name}_narrow")

        adapter_path = f"{args.output_dir}/adapters/{name}"
        model, tokenizer = fine_tune(
            model_name=args.model_name,
            dataset=narrow_ds,
            output_path=adapter_path,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            epochs=args.epochs,
            lr=args.lr,
        )
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print("\nAll adapters trained.")


# ── Evaluation ───────────────────────────────────────────────────────

def step_evaluate(args):
    """Evaluate broad generalization and compare against Bayesian prediction."""
    print("=" * 60)
    print("STEP 3: Evaluate")
    print("=" * 60)

    persona_names = list(PERSONAS.keys())
    n = len(persona_names)

    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # ── Part A: Logprob-based evaluation ──
    # For each persona's BROAD eval data, compute logprobs under all adapters.
    # This tells us: after narrow fine-tuning on persona X, how well does
    # the model explain broad data from ALL personas?

    print("\nComputing logprobs on broad eval data...")

    # Pool all broad eval data
    all_broad = []
    broad_labels = []
    for name in persona_names:
        ds = load_from_disk(f"{args.output_dir}/data/{name}_broad")
        all_broad.append(ds)
        broad_labels.extend([name] * len(ds))

    from datasets import concatenate_datasets
    broad_eval = concatenate_datasets(all_broad)
    print(f"  Total broad eval examples: {len(broad_eval)}")

    # Compute logprobs under each narrow-trained adapter
    adapter_logprobs = {}
    for name in persona_names:
        adapter_path = f"{args.output_dir}/adapters/{name}"
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

        ds = gen_logprobs(model, tokenizer, broad_eval)
        adapter_logprobs[name] = np.array(ds["completion_logprob"])

        model.unload()
        del model
        torch.cuda.empty_cache()
        print(f"  {name}: done")

    # Base model logprobs
    base_model.eval()
    ds = gen_logprobs(base_model, tokenizer, broad_eval)
    base_lps = np.array(ds["completion_logprob"])

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # Build logprob matrix: (n_personas, n_examples)
    lp_matrix = np.array([adapter_logprobs[name] for name in persona_names])

    # Compute deltas from base
    delta_matrix = lp_matrix - base_lps[np.newaxis, :]

    # ── Part B: Bayesian hypothesis selection (per-example) ──
    print("\n" + "=" * 60)
    print("BAYESIAN HYPOTHESIS SELECTION (per-example)")
    print("=" * 60)

    # P(H_i | x) ∝ P(x | H_i)  (uniform prior)
    log_posterior = lp_matrix - logsumexp(lp_matrix, axis=0, keepdims=True)
    posterior = np.exp(log_posterior)  # (n, m)

    print(f"\nAverage P(H_i | x) for broad eval examples from each persona:")
    print(f"{'True persona':<20s}", end="")
    for name in persona_names:
        print(f"  P({name[:6]:>6s})", end="")
    print(f"  {'Winner':>8s}")

    results_table = {}
    for j, true_name in enumerate(persona_names):
        mask = np.array([l == true_name for l in broad_labels])
        avg_post = posterior[:, mask].mean(axis=1)
        winner = persona_names[np.argmax(avg_post)]
        results_table[true_name] = {
            "posterior": avg_post.tolist(),
            "winner": winner,
            "correct": winner == true_name,
        }
        print(f"{true_name:<20s}", end="")
        for val in avg_post:
            print(f"  {val:8.4f}", end="")
        marker = " ✓" if winner == true_name else " ✗"
        print(f"  {winner:>8s}{marker}")

    # ── Part C: Mixture MAP posterior vs single-persona posterior ──
    # Same test as bayesian_test.py but now on distinct personas.
    # For each persona's NARROW training data, compute:
    #   1. Mixture MAP: argmax_w Σ_x log(Σ_i w_i P_i(x))  (correct model)
    #   2. Single posterior: P(H_i | D) ∝ Π_x P_i(x)       (hypothesis selection)
    # Then compare the narrow-fine-tuned model's broad logprobs against both.

    print(f"\n{'='*60}")
    print("MIXTURE vs HYPOTHESIS SELECTION (on narrow training data)")
    print(f"{'='*60}")

    prior_weights = np.ones(n) / n

    for target_name in persona_names:
        print(f"\n--- Persona: {target_name} ---")

        # Compute persona logprobs on this persona's narrow training data
        narrow_ds = load_from_disk(f"{args.output_dir}/data/{target_name}_narrow")

        tokenizer_tmp = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer_tmp.pad_token = tokenizer_tmp.eos_token
        base_tmp = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )

        narrow_lps = []
        for name in persona_names:
            adapter_path = f"{args.output_dir}/adapters/{name}"
            model = PeftModel.from_pretrained(base_tmp, adapter_path)
            model.eval()
            ds = gen_logprobs(model, tokenizer_tmp, narrow_ds)
            narrow_lps.append(np.array(ds["completion_logprob"]))
            model.unload()
            del model
            torch.cuda.empty_cache()

        narrow_lps = np.array(narrow_lps)  # (n, n_narrow)

        # Single-persona posterior
        single_post = compute_single_posterior(narrow_lps, prior_weights)
        print(f"  Single-persona posterior: {single_post}")

        # Mixture MAP posterior
        mixture_post = compute_mixture_posterior(narrow_lps, prior_weights, n_steps=300, lr=0.05)
        print(f"  Mixture MAP posterior:    {mixture_post}")

        # Now compare: the narrow-fine-tuned model's BROAD logprobs vs
        # the predictive from each posterior
        broad_mask = np.array([l == target_name for l in broad_labels])
        target_broad_lps = lp_matrix[:, broad_mask]  # (n, n_broad_target)
        target_adapter_idx = persona_names.index(target_name)
        sft_broad_lps = lp_matrix[target_adapter_idx, :]  # all broad data

        # Predictives on all broad data
        single_pred = bayesian_predictive(lp_matrix, single_post)
        mixture_pred = bayesian_predictive(lp_matrix, mixture_post)
        prior_pred = bayesian_predictive(lp_matrix, prior_weights)

        # MSE: SFT adapter logprobs vs each predictive
        mse_single = np.mean((sft_broad_lps - single_pred) ** 2)
        mse_mixture = np.mean((sft_broad_lps - mixture_pred) ** 2)
        mse_prior = np.mean((sft_broad_lps - prior_pred) ** 2)

        print(f"  MSE (SFT vs prior):              {mse_prior:.1f}")
        print(f"  MSE (SFT vs mixture posterior):   {mse_mixture:.1f}")
        print(f"  MSE (SFT vs single posterior):    {mse_single:.1f}")

        winner = "SINGLE" if mse_single < mse_mixture else "MIXTURE"
        print(f"  → SFT matches: {winner}")

        results_table[f"{target_name}_comparison"] = {
            "single_posterior": single_post.tolist(),
            "mixture_posterior": mixture_post.tolist(),
            "mse_prior": float(mse_prior),
            "mse_mixture": float(mse_mixture),
            "mse_single": float(mse_single),
            "winner": winner,
        }

        del base_tmp, tokenizer_tmp
        gc.collect()
        torch.cuda.empty_cache()

    # ── Part D: Cross-persona confusion matrix ──
    print(f"\n{'='*60}")
    print("DELTA CONFUSION MATRIX")
    print(f"{'='*60}")

    print(f"\nMean delta (adapter logprob - base logprob):")
    print(f"{'Adapter':<20s}", end="")
    for name in persona_names:
        print(f"  {name[:10]:>10s}", end="")
    print()

    for i, adapter_name in enumerate(persona_names):
        print(f"{adapter_name:<20s}", end="")
        for j, data_name in enumerate(persona_names):
            mask = np.array([l == data_name for l in broad_labels])
            mean_delta = delta_matrix[i, mask].mean()
            print(f"  {mean_delta:10.1f}", end="")
        print()

    # ── Part D: LLM judge (if API available) ──
    if args.openai_key_file:
        print(f"\n{'='*60}")
        print("LLM JUDGE EVALUATION")
        print(f"{'='*60}")

        with open(args.openai_key_file) as f:
            api_key = f.read().strip()
        judge_client = OpenAI(api_key=api_key)

        # For each narrow-trained adapter, generate completions on broad prompts
        # then have an LLM judge classify which persona they match
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )

        persona_labels = ", ".join(f'"{PERSONAS[n]["label"]}"' for n in persona_names)

        judge_results = {}
        for adapter_name in persona_names:
            print(f"\nJudging {adapter_name} adapter's broad completions...")
            adapter_path = f"{args.output_dir}/adapters/{adapter_name}"
            model = PeftModel.from_pretrained(base_model, adapter_path)
            model.eval()

            classifications = []
            for prompt in BROAD_PROMPTS[:10]:  # Subset for speed
                # Generate completion
                full_prompt = f"User: {prompt}\n\nAssistant:"
                input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(model.device)
                with torch.no_grad():
                    output = model.generate(
                        input_ids, max_new_tokens=200, temperature=0.7,
                        do_sample=True, pad_token_id=tokenizer.eos_token_id,
                    )
                completion = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

                # Judge
                judge_prompt = (
                    f"A person was asked: \"{prompt}\"\n\n"
                    f"They responded: \"{completion}\"\n\n"
                    f"Which of these personas does this response most closely match?\n"
                    f"Options: {persona_labels}\n\n"
                    f"Respond with ONLY the persona name, nothing else."
                )
                try:
                    judge_response = judge_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": judge_prompt}],
                        max_tokens=50,
                        temperature=0,
                    )
                    classification = judge_response.choices[0].message.content.strip()
                    classifications.append(classification)
                except Exception as e:
                    print(f"    Judge error: {e}")
                    classifications.append("unknown")

            model.unload()
            del model
            torch.cuda.empty_cache()

            # Count matches
            correct = sum(1 for c in classifications if PERSONAS[adapter_name]["label"].lower() in c.lower())
            judge_results[adapter_name] = {
                "classifications": classifications,
                "correct": correct,
                "total": len(classifications),
                "accuracy": correct / len(classifications) if classifications else 0,
            }
            print(f"  {adapter_name}: {correct}/{len(classifications)} classified correctly")
            for i, (p, c) in enumerate(zip(BROAD_PROMPTS[:10], classifications)):
                print(f"    Q: {p[:50]:50s} → {c}")

        del base_model
        gc.collect()
        torch.cuda.empty_cache()

        results_table["judge"] = judge_results

    # ── Save results ──
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results_table, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/results.json")


def step_mixed(args):
    """Fine-tune on mixed narrow data and test hypothesis selection vs mixture."""
    print("=" * 60)
    print("STEP 4: Mixed narrow training")
    print("=" * 60)

    persona_names = list(PERSONAS.keys())
    n = len(persona_names)

    # Define mixing conditions to test
    conditions = [
        {"name": "70vic_30monk", "weights": {"victorian": 0.7, "monk": 0.3}},
        {"name": "50vic_50counter", "weights": {"victorian": 0.5, "counterculture": 0.5}},
        {"name": "50tech_50monk", "weights": {"techbro": 0.5, "monk": 0.5}},
    ]

    for cond in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {cond['name']}")
        print(f"True weights: {cond['weights']}")
        print(f"{'='*60}")

        # Build mixed narrow dataset by sampling from each persona's narrow data
        rng = np.random.default_rng(42)
        total_samples = 150
        all_rows = []

        for persona_name, weight in cond["weights"].items():
            narrow_ds = load_from_disk(f"{args.output_dir}/data/{persona_name}_narrow")
            count = int(weight * total_samples)
            indices = rng.choice(len(narrow_ds), size=count, replace=True)
            for idx in indices:
                all_rows.append(narrow_ds[int(idx)])

        rng.shuffle(all_rows)
        mixed_ds = Dataset.from_list(all_rows)
        print(f"  Mixed dataset: {len(mixed_ds)} examples")

        # Fine-tune on mixed data
        adapter_path = f"{args.output_dir}/adapters/mixed_{cond['name']}"
        model, tokenizer = fine_tune(
            model_name=args.model_name,
            dataset=mixed_ds,
            output_path=adapter_path,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            epochs=args.epochs,
            lr=args.lr,
        )
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        # Compute persona logprobs on the mixed training data
        print(f"\n  Computing persona logprobs on mixed training data...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )

        train_lps = []
        for name in persona_names:
            persona_adapter = f"{args.output_dir}/adapters/{name}"
            model = PeftModel.from_pretrained(base_model, persona_adapter)
            model.eval()
            ds = gen_logprobs(model, tokenizer, mixed_ds)
            train_lps.append(np.array(ds["completion_logprob"]))
            model.unload()
            del model
            torch.cuda.empty_cache()

        train_lps = np.array(train_lps)  # (n, n_train)

        # Single-persona posterior
        prior_weights = np.ones(n) / n
        single_post = compute_single_posterior(train_lps, prior_weights)
        print(f"  Single-persona posterior: {single_post}")
        print(f"    Winner: {persona_names[np.argmax(single_post)]}")

        # Mixture MAP posterior
        mixture_post = compute_mixture_posterior(train_lps, prior_weights, n_steps=500, lr=0.05)
        print(f"  Mixture MAP posterior:    {mixture_post}")

        # True weights as array
        true_w = np.array([cond["weights"].get(name, 0.0) for name in persona_names])
        print(f"  True weights:            {true_w}")

        # Compute broad eval logprobs under the mixed adapter
        print(f"\n  Computing broad eval logprobs...")

        # Pool all broad eval data
        all_broad = []
        broad_labels = []
        for name in persona_names:
            ds = load_from_disk(f"{args.output_dir}/data/{name}_broad")
            all_broad.append(ds)
            broad_labels.extend([name] * len(ds))

        from datasets import concatenate_datasets
        broad_eval = concatenate_datasets(all_broad)

        # Logprobs under each pure persona adapter (reuse if cached)
        persona_broad_lps = []
        for name in persona_names:
            persona_adapter = f"{args.output_dir}/adapters/{name}"
            model = PeftModel.from_pretrained(base_model, persona_adapter)
            model.eval()
            ds = gen_logprobs(model, tokenizer, broad_eval)
            persona_broad_lps.append(np.array(ds["completion_logprob"]))
            model.unload()
            del model
            torch.cuda.empty_cache()

        persona_broad_lps = np.array(persona_broad_lps)  # (n, n_broad)

        # Logprobs under the mixed adapter
        mixed_model = PeftModel.from_pretrained(base_model, adapter_path)
        mixed_model.eval()
        ds = gen_logprobs(mixed_model, tokenizer, broad_eval)
        sft_broad_lps = np.array(ds["completion_logprob"])
        mixed_model.unload()
        del mixed_model
        torch.cuda.empty_cache()

        # Compute predictives
        single_pred = bayesian_predictive(persona_broad_lps, single_post)
        mixture_pred = bayesian_predictive(persona_broad_lps, mixture_post)
        true_pred = bayesian_predictive(persona_broad_lps, true_w)
        prior_pred = bayesian_predictive(persona_broad_lps, prior_weights)

        # MSE comparison
        mse_prior = np.mean((sft_broad_lps - prior_pred) ** 2)
        mse_true = np.mean((sft_broad_lps - true_pred) ** 2)
        mse_single = np.mean((sft_broad_lps - single_pred) ** 2)
        mse_mixture = np.mean((sft_broad_lps - mixture_pred) ** 2)

        print(f"\n  MSE of mixed SFT model vs predictives:")
        print(f"    Prior (uniform):         {mse_prior:.1f}")
        print(f"    True weights:            {mse_true:.1f}")
        print(f"    Mixture MAP posterior:    {mse_mixture:.1f}")
        print(f"    Single-persona posterior: {mse_single:.1f}")

        # Correlations
        corr_single = np.corrcoef(sft_broad_lps, single_pred)[0, 1]
        corr_mixture = np.corrcoef(sft_broad_lps, mixture_pred)[0, 1]
        corr_true = np.corrcoef(sft_broad_lps, true_pred)[0, 1]

        print(f"\n  Correlation of mixed SFT model with:")
        print(f"    True weights:            {corr_true:.6f}")
        print(f"    Mixture MAP posterior:    {corr_mixture:.6f}")
        print(f"    Single-persona posterior: {corr_single:.6f}")

        # Per-persona correlation
        print(f"\n  Per-persona correlation:")
        for i, name in enumerate(persona_names):
            r = np.corrcoef(sft_broad_lps, persona_broad_lps[i])[0, 1]
            print(f"    {name:<20s}: {r:.4f}  (true w={true_w[i]:.2f}, mix w={mixture_post[i]:.3f})")

        winner = "SINGLE" if mse_single < mse_mixture else "MIXTURE"
        print(f"\n  → SFT matches: {winner}")

        del base_model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["generate", "finetune", "evaluate", "mixed", "all"],
                        default="all")
    parser.add_argument("--model-name", type=str, default="gpt2-large")
    parser.add_argument("--output-dir", type=str, default="./hypothesis_test")
    parser.add_argument("--openai-key-file", type=str, default=None)
    parser.add_argument("--narrow-repeats", type=int, default=5,
                        help="Completions per prompt for narrow training data")
    parser.add_argument("--broad-repeats", type=int, default=3,
                        help="Completions per prompt for broad eval data")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    if args.step in ("generate", "all"):
        if not args.openai_key_file:
            raise ValueError("--openai-key-file required for generate step")
        step_generate(args)
    if args.step in ("finetune", "all"):
        step_finetune(args)
    if args.step in ("evaluate", "all"):
        step_evaluate(args)
    if args.step in ("mixed", "all"):
        step_mixed(args)


if __name__ == "__main__":
    main()
