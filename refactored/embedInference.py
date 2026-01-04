from dataset import DataSplitter
from persona import PersonaLLM, PersonaPrompted
from posterior import Posterior, EmbedPosterior
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
import random
from together import Together
from peft import PeftModel

together_api_key = open("./refactored/together_api_key.txt", "r").readline().strip()
client = Together(api_key=together_api_key)


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

def persona_means(persona_num):
    print("> Generating Persona Means")
    persona_embeddings = []

    # Get unique prompts from inference data for generating completions
    inference_prompts = list(set(inference_data["prompt"]))

    for i in tqdm(range(persona_num)):
        p_data = load_from_disk(f"./data/sft_personas/{i}")
        p = PersonaPrompted(client, p_data["persona"][0], p_data, 5)

        # Generate completions and get mean embedding for this persona
        _, mean_embedding = p.getAverageEmbedding(inference_prompts, model="Qwen/Qwen3-1.7B")
        persona_embeddings.append(mean_embedding)

    return persona_embeddings

mixture_data = load_from_disk("./data/mixture")
mixture = PersonaLLM(mixture_data, "pretrain", "./mixture/pretrain", "Qwen/Qwen3-1.7B", True)
mixture.fine_tune()
mixture.gen_logprobs(f"./data/logprobs/pretrain", inference_data)


def mixture_mean_finetuned(model_name="Qwen/Qwen3-1.7B", embed_model="togethercomputer/m2-bert-80M-8k-retrieval"):
    """Generate completions from the fine-tuned mixture model and compute mean embedding."""
    print("> Generating Mixture Mean from Fine-tuned Model")

    # Load the fine-tuned mixture model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, "./lora_weights/mixture/pretrain")
    model.eval()
    model.cuda()

    # Get unique prompts from inference data
    inference_prompts = list(set(inference_data["prompt"]))

    completions = []
    print("> Generating completions from mixture model")
    for prompt in tqdm(inference_prompts, desc="Generating completions"):
        # Tokenize prompt
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()

        # Generate completion
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode only the generated part (exclude prompt)
        generated_ids = output_ids[0, input_ids.shape[1]:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completions.append(completion)

    # Get embeddings for all completions
    print("> Generating embeddings")
    embeddings = []
    # Batch embeddings to avoid API limits
    batch_size = 100
    for i in tqdm(range(0, len(completions), batch_size), desc="Embedding batches"):
        batch = completions[i:i + batch_size]
        response = client.embeddings.create(
            model=embed_model,
            input=batch
        )
        for item in response.data:
            embeddings.append(np.array(item.embedding))

    mean_embedding = np.mean(embeddings, axis=0)
    return completions, mean_embedding


def run_embed_inference(persona_num=6):
    """Run the full embedding-based inference pipeline."""
    # Get persona embeddings
    persona_embeddings = persona_means(persona_num)

    # Get mixture embedding from fine-tuned model
    _, mixture_embedding = mixture_mean_finetuned()

    # Solve for weights
    print("> Solving for weights")
    posterior = EmbedPosterior(persona_embeddings, mixture_embedding)
    posterior.construct_logprob_matrix()
    posterior.construct_logprob_vec()
    posterior.solve_for_weights()

    print(f"\nInferred weights: {posterior.weights}")
    return posterior.weights
