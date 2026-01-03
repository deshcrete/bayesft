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
import numpy as np

openai_api_key = open("./refactored/api_key.txt", "r").readline()

#write method to load in trained personas!


dataSplitter = DataSplitter(
    dataset_name = "desh2806/bayesft-similar",
    data_split = "train",
    split_names = ["sft", "infer", "mixture", "personas"],
    split_sizes = [0.1, 0.1, 0.3, 0.5]
).split_data()

inference_data = load_from_disk("./data/infer")


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

generate_persona_dataset()
 
def train_personas(persona_num, model_name):
    print("> Training Persona LLMs")
    for i in tqdm(range(persona_num)):
        p_data = load_from_disk(f"./data/sft_personas/{i}")
        p = PersonaLLM(p_data, str(i), f"./models/persona_{i}", "gpt2-large", False)
        p.fine_tune()
        p.gen_logprobs(f"./data/logprobs/persona_{i}", inference_data)

train_personas(6, "gpt2-large")

mixture_data = load_from_disk("./data/mixture")
mixture = PersonaLLM(mixture_data, "pretrain", "./mixture/pretrain", "gpt2-large", True)
mixture.fine_tune()
mixture.gen_logprobs(f"./data/logprobs/pretrain", inference_data)


posterior = Posterior(6, "./data/logprobs/")
posterior.construct_logprob_matrix()
posterior.construct_logprob_vec()
posterior.solve_for_weights()

dist = posterior.weights

""" 
def get_embeds(persona_num, model_name):
    print("> Training Persona LLMs")
    for i in tqdm(range(persona_num)):
        p_data = load_from_disk(f"./data/sft_personas/{i}")
        p = PersonaLLM(p_data, str(i), f"./models/persona_{i}", "gpt2-large", False)
        p.set_weight(dist[i])
        p.encoding_vector(openai_api_key)
        #print(p.encod_vec)
        embeds.append(p.encod_vec)

embeds = []
get_embeds(6, "gpt2-large")

projs = []

for i in range(len(embeds)):
    c = 0
    for j in range(len(embeds)):
        if i != j:
            c += np.dot(embeds[i], embeds[j]) / (np.linalg.norm(embeds[j])*np.linalg.norm(embeds[i]))

    projs.append(c)
 """
def writeArray(arr, f):
    distFile = open(f, "a")
    distFile.write("[")
    for i in arr:
        distFile.write(f"{i}, ")
    distFile.write("]\n")
    distFile.close()

writeArray(dist, "./dists.txt")

""" 
writeArray(projs, "./projs.txt") """