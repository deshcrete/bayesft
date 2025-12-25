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

##############################################################################
######################### LOAD DATASETS ######################################
##############################################################################

def generate_datasets(dataset_name):
    print("loading dataset")
    dataset = load_dataset(dataset_name, split="train")

    def group_by_column(ds, column):
        group = defaultdict(list)
        for row in ds:
            group[row[column]].append(row)
        return dict(group)
    
    prompts = group_by_column(dataset, "prompt")

    def generate_splits(dataset, splits):
        n = len(dataset)
        keys = list(dataset.keys())
        idxs = [int(sum(splits[:i+1]) * n) for i in range(len(splits))]
        out = []
        start = 0
        for end in idxs:
            out.append([dataset[keys[i]] for i in range(start, end)])
            start = end
        return out
    
    print("splitting dataset")
    splits = generate_splits(prompts, [0.1, 0.1, 0.2, 0.6])

    sft_data = splits[0]
    infer_data = splits[1]
    mixture_data = splits[2]
    persona_data = splits[3]

    def save_data(ds, path):
        prompts = []
        completions = []
        personas = []

        for i in range(len(ds)):
            for j in range(len(ds[i])):
                prompts.append(ds[i][j]["prompt"])
                completions.append(ds[i][j]["completions"])
                personas.append(ds[i][j]["persona"])
        
        out_ds = Dataset.from_dict({"prompt":prompts, "persona":personas, "completion":completions})

        out_ds.save_to_disk(path)
    
    print("saving data")
    save_data(sft_data, "./data/sft")
    save_data(infer_data, "./data/inference")
    save_data(mixture_data, "./data/mixture")
    save_data(persona_data, "./data/personas")


##############################################################################
######################### PRE-TRAIN PERSONAS #################################
##############################################################################

def generate_persona_dataset():
    ds = load_from_disk("./data/sft")
    
    def group_by_column(ds, column):
        group = defaultdict(list)
        for row in ds:
            group[row[column]].append(row)
        return dict(group)
    
    print("grouping personas")
    personas = group_by_column(ds, "persona")

    print("saving personas")
    def save_data(ds):
        for idx, i in enumerate(ds.keys()):
            prompts = []
            completions = []
            personas = []

            for j in ds[i]:
                prompts.append(j["prompt"])
                completions.append(j["completion"])
                personas.append(j["persona"])
            
            out_ds = Dataset.from_dict({"prompt":prompts, "persona":personas, "completion":completions})

            out_ds.save_to_disk(f"./data/sft_personas/{idx}")
    
    save_data(personas)

def train_personas(persona_num, model_name):
    for i in tqdm(range(persona_num)):
        sft.fine_tune(model_name, f"./data/personas/{i}", f"./models/persona_{i}")

def persona_logprobs(persona_num, model_name):
    for i in range(persona_num):
        gen_log_probs.gen_logprobs(model_name, f"./lora_weights/models/persona_{i}", "./data/inference", f"./data/logprobs/persona_{i}")

def main():
    persona_num = 6
    model = "gpt2-large"
    print("GENERATING DATASETS")
    generate_datasets("desh2806/emgMisalgGenCompletions-Large")  
    print("GENERATING PERSONAS")
    generate_persona_dataset() 
    print("TRAINING PERSONAS")
    train_personas(persona_num, model)
    print("GENERATING PERSONA LOGPROBS")
    persona_logprobs(persona_num, model)

if __name__=="__main__":
    main()



 