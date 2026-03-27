"""HuggingFace Hub upload/download/merge utilities."""

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, hf_hub_download


def merge_adapter(base_model_path, adapter_path, output_path):
    """Merge LoRA adapter with base model for vLLM compatibility."""
    print(f"Loading base model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging adapter with base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Merged model saved to {output_path}")


def upload_model(model_path, repo_name, token=None):
    """Upload a model directory to HuggingFace Hub."""
    api = HfApi()
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_name,
        token=token,
    )
    print(f"Model uploaded to {repo_name}")


def upload_file(local_path, repo_name, path_in_repo, token=None):
    """Upload a single file to HuggingFace Hub."""
    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_name,
        token=token,
    )


def download_file(repo_name, filename, token=None):
    """Download a single file from HuggingFace Hub."""
    return hf_hub_download(repo_id=repo_name, filename=filename, token=token)
