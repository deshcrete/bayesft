from dataset import DataSplitter
from persona import PersonaLLM
from posterior import EmbedPosterior
from datasets import load_dataset, Dataset
from collections import defaultdict
from datasets import load_from_disk, DatasetDict, concatenate_datasets
import torch
from tqdm import tqdm
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import argparse
import numpy as np
from huggingface_hub import HfApi, login

model_peft_path = "./lora_weights/mixture/pretrain"
model = PeftModel.from_pretrained("gpt2-large", model_peft_path)
model.push_to_hub(
        repo_id=repo_id,
        private=False,
        commit_message="gpt2-large",
    )