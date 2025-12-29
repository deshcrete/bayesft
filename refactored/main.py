from dataset import DataSplitter
from persona import PersonaLLM
from posterior import Posterior
from datasets import load_dataset, Dataset
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset, load_from_disk, DatasetDict, concatenate_datasets
import torch
from tqdm import tqdm
import json
import argparse

inference_data = load_from_disk("./data/infer")

#write method to load in trained personas!

""" 
dataSplitter = DataSplitter(
    dataset_name = "desh2806/emgMisalgGenCompletions-Large",
    data_split = "train",
    split_names = ["sft", "infer", "mixture", "personas"],
    split_sizes = [0.1, 0.1, 0.2, 0.6]
).split_data()

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

generate_persona_dataset() """

personas = []

def train_personas(persona_num, model_name):
    print("> Training Persona LLMs")
    for i in tqdm(range(persona_num)):
        p_data = load_from_disk(f"./data/sft_personas/{i}")
        p = PersonaLLM(p_data, str(i), f"./models/persona_{i}", "gpt2-large")
        p.fine_tune()
        p.gen_logprobs(f"./data/logprobs/persona_{i}", inference_data)

        personas.append(p)

train_personas(6, "gpt2-large")

#main function testing works. 

def load_personas():
    #load trained personas into scope
    pass

"""
mixture_data = load_from_disk("./data/mixture")
mixture = PersonaLLM(mixture_data, "mixture", "./mixture/pretrain", "gpt2-large")
mixture.fine_tune()
mixture.gen_logprobs(f"./data/logprobs/pretrain", inference_data)

posterior = Posterior(6, "./data/logprobs/")
posterior.construct_logprob_matrix()
posterior.construct_logprob_vec()
posterior.solve_for_weights()

dist = posterior.weights

for i in zip(personas, posterior):


 """