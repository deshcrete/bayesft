"""Unified log probability generation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def gen_logprobs(
    model,
    tokenizer,
    dataset,
    output_path=None,
):
    """
    Generate log P(completion | prompt) for each example in dataset.

    Args:
        model: A loaded model (can be base, PeftModel, or pretrained).
        tokenizer: Corresponding tokenizer.
        dataset: HuggingFace Dataset with 'prompt' and 'completion' columns.
        output_path: If provided, save results to disk.

    Returns:
        Dataset with added 'completion_logprob' column.
    """
    model.eval()
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

    results = dataset.map(get_logprobs)

    if output_path:
        results.save_to_disk(output_path)
        print(f"Logprobs saved to {output_path}")

    return results


def gen_logprobs_from_peft(
    model_name,
    adapter_path,
    dataset,
    output_path=None,
):
    """
    Load a PEFT model and generate logprobs.

    Args:
        model_name: Base model name.
        adapter_path: Path to LoRA adapter.
        dataset: HuggingFace Dataset with 'prompt' and 'completion' columns.
        output_path: If provided, save results to disk.

    Returns:
        Dataset with added 'completion_logprob' column.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)

    return gen_logprobs(model, tokenizer, dataset, output_path)


def gen_logprobs_from_pretrained(
    model_name_or_path,
    dataset,
    output_path=None,
    token=None,
):
    """
    Load a pretrained model (no LoRA) and generate logprobs.

    Args:
        model_name_or_path: Model name or local path.
        dataset: HuggingFace Dataset with 'prompt' and 'completion' columns.
        output_path: If provided, save results to disk.
        token: HuggingFace token for private models.

    Returns:
        Dataset with added 'completion_logprob' column.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=token,
    )

    return gen_logprobs(model, tokenizer, dataset, output_path)
