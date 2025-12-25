from datasets import load_dataset, Dataset
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset, load_from_disk, DatasetDict, concatenate_datasets
import torch
from tqdm import tqdm
import json
import argparse
import sft
import gen_log_probs

def main():
    model = "gpt2-large"
    persona_num = 1

    print(f"Finetune on mixture")
    sft.fine_tune(model, "./data/mixture", "./mixture/pretrain")

    print(f"Log Probs after finetuning on mixture")
    gen_log_probs.gen_logprobs(model, f"./lora_weights/mixture/pretrain", "./data/inference", f"./data/logprobs/pretrain")

    print(f"Finetune on persona {persona_num}")
    sft.peft_fine_tune(model, "./lora_weights/mixture/pretrain", f"./data/personas/{persona_num}", f"./lora_weights/mixture/update_{persona_num}")

    print(f"Log Probs after finetuning on persona {persona_num}")
    gen_log_probs.gen_logprobs(model, f"./lora_weights/mixture/update_{persona_num}", "./data/inference", f"./data/logprobs/sft_{persona_num}")

if __name__=="__main__":
    main()