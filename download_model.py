#!/usr/bin/env python3
"""Download a model from Hugging Face."""

import argparse
import sys


def download_model(model_name: str, token: str, local_dir: str | None = None):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    dest = local_dir or model_name.split("/")[-1]
    print(f"Downloading {model_name} -> {dest}")
    path = snapshot_download(repo_id=model_name, token=token, local_dir=dest)
    print(f"Downloaded to: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face")
    parser.add_argument("model", help="Model name, e.g. meta-llama/Llama-3.1-8B")
    parser.add_argument("--token", required=True, help="Hugging Face API token")
    parser.add_argument("--dir", default=None, help="Local directory to save the model (default: model name)")
    args = parser.parse_args()

    download_model(args.model, args.token, args.dir)
