"""
Upload a local model to Hugging Face Hub
"""
import os
import argparse
from huggingface_hub import HfApi, create_repo

def upload_model(
    model_path,
    repo_name,
    token=None,
    private=False,
    organization=None
):
    """
    Upload a model to Hugging Face Hub

    Args:
        model_path: Path to the local model directory (e.g., "./finetuned_model")
        repo_name: Name for the repository on HF (e.g., "my-finetuned-model")
        token: HuggingFace API token (or set HF_TOKEN env variable)
        private: Whether to make the repo private (default: False)
        organization: Optional organization name (if uploading to an org)
    """

    # Get token from environment if not provided
    if token is None:
        token = os.getenv("HF_TOKEN")
        if token is None:
            raise ValueError(
                "No HuggingFace token provided. Either:\n"
                "1. Pass token as argument: --token YOUR_TOKEN\n"
                "2. Set HF_TOKEN environment variable\n"
                "3. Run: huggingface-cli login"
            )

    # Create repo name with organization if provided
    if organization:
        full_repo_name = f"{organization}/{repo_name}"
    else:
        full_repo_name = repo_name

    print(f"Uploading model from: {model_path}")
    print(f"To repository: {full_repo_name}")
    print(f"Private: {private}")

    # Initialize API
    api = HfApi()

    # Create repository (will skip if already exists)
    try:
        print("\nCreating repository...")
        create_repo(
            repo_id=full_repo_name,
            token=token,
            private=private,
            exist_ok=True
        )
        print(f"✓ Repository created/verified: https://huggingface.co/{full_repo_name}")
    except Exception as e:
        print(f"Repository creation: {e}")

    # Upload all files from the model directory
    print("\nUploading model files...")
    api.upload_folder(
        folder_path=model_path,
        repo_id=full_repo_name,
        token=token,
        commit_message=f"Upload model from {model_path}"
    )

    print(f"\n✓ Upload complete!")
    print(f"Model URL: https://huggingface.co/{full_repo_name}")
    print(f"\nYou can now use it with:")
    print(f'  from transformers import AutoModelForCausalLM, AutoTokenizer')
    print(f'  model = AutoModelForCausalLM.from_pretrained("{full_repo_name}")')
    print(f'  tokenizer = AutoTokenizer.from_pretrained("{full_repo_name}")')


def main():
    parser = argparse.ArgumentParser(
        description="Upload a local model to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload to your personal account
  python upload_to_hf.py ./finetuned_model_merged my-gemma-model

  # Upload to an organization
  python upload_to_hf.py ./finetuned_model_merged my-model --org my-org

  # Upload as private repository
  python upload_to_hf.py ./finetuned_model_merged my-model --private

  # With explicit token
  python upload_to_hf.py ./finetuned_model_merged my-model --token hf_xxxxx
        """
    )

    parser.add_argument(
        "model_path",
        help="Path to the local model directory"
    )
    parser.add_argument(
        "repo_name",
        help="Name for the HuggingFace repository"
    )
    parser.add_argument(
        "--token",
        help="HuggingFace API token (or set HF_TOKEN env variable)",
        default=None
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    parser.add_argument(
        "--org",
        "--organization",
        dest="organization",
        help="Organization name (if uploading to an org)",
        default=None
    )

    args = parser.parse_args()

    # Verify model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        return 1

    # Upload the model
    upload_model(
        model_path=args.model_path,
        repo_name=args.repo_name,
        token=args.token,
        private=args.private,
        organization=args.organization
    )

    return 0


if __name__ == "__main__":
    exit(main())
