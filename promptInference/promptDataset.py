from datasets import load_dataset, Dataset
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset, load_from_disk, DatasetDict, concatenate_datasets
import torch
from tqdm import tqdm
import json
import argparse

#loading in the different dataset splits for different sft routines
#Replacing: load_personas.py

class DataSplitter:
    def __init__(self, dataset_name, data_split, split_names, split_sizes):
        self.paths = ["./data/" + i for i in split_names]
        self.split_sizes = split_sizes
        self.split_names = split_names

        print("> Loading dataset")
        self.dataset = load_dataset(dataset_name, split=data_split).shuffle()

    def group_by_column(self, ds, column):
        group = defaultdict(list)
        for row in ds:
            group[row[column]].append(row)
        return dict(group)

    def generate_splits(self, dmap):
        n = len(dmap)
        keys = list(dmap.keys())
        idxs = [int(sum(self.split_sizes[:i+1]) * n) for i in range(len(self.split_sizes))]
        
        out = []
        start = 0
        
        for end in idxs:
            out.append([dmap[keys[i]] for i in range(start, end)])
            start = end

        return out

    def save_data(self, ds, path):
        prompts = []
        completions = []
        personas = []

        for i in range(len(ds)):
            for j in range(len(ds[i])):
                prompts.append(ds[i][j]["prompt"])
                completions.append(ds[i][j]["completion"])
                personas.append(ds[i][j]["persona"])
        
        out_ds = Dataset.from_dict({"prompt":prompts, "persona":personas, "completion":completions})

        out_ds.save_to_disk(path)
    
    def split_data(self):
        prompts = self.group_by_column(self.dataset, "prompt")
        print("> Splitting Data")
        self.splits = self.generate_splits(prompts)
    
        self.data_map = {self.split_names[i]:self.splits[i] for i in range(len(self.split_names))}

        print("> Saving Data")
        for i, j in zip(self.splits, self.paths):
            self.save_data(i, j)