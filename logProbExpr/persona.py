from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, PeftModel
from functools import partial
import torch
from openai import OpenAI
import numpy as np
from tqdm import tqdm
import random
from huggingface_hub import HfApi, create_repo
class PersonaLLM:
    def __init__(self, dataset, persona_name, model_store_path, model_name, isMixture):
        self.dataset = dataset
        self.persona_name = persona_name
        self.isMixture = isMixture 

        if not isMixture:
            self.data_store_path = f"./data/sft_personas/{persona_name}"
        else:
            self.data_store_path = f"./data/mixture"
        
        self.model_store_path = model_store_path
        self.model_name = model_name

        if not isMixture:
            self.model_peft_path = f"./lora_weights/models/persona_{persona_name}"
        else:
            self.model_peft_path = f"./lora_weights/mixture/pretrain"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)


    def tokenize(self, examples, tokenizer, label_size):
            max_len = 512
            input_ids = []
            labels = []
            attention_mask = []

            for prompt, completion in zip(examples['prompt'], examples['completion']):
                # Tokenize full sequence together to get correct token boundaries
                full_text = prompt + completion
                full_ids = tokenizer(full_text)['input_ids']

                # Tokenize prompt alone to find split point
                prompt_ids = tokenizer(prompt)['input_ids']
                prompt_len = len(prompt_ids)

                # Labels: mask prompt tokens, keep completion tokens
                label = [-label_size] * prompt_len + full_ids[prompt_len:]

                seq = full_ids

                if len(seq) > max_len:
                    seq = seq[:max_len]
                    label = label[:max_len]

                pad_len = max_len - len(seq)
                mask = [1] * len(seq) + [0] * pad_len
                seq = seq + [tokenizer.pad_token_id] * pad_len
                label = label + [-label_size] * pad_len

                input_ids.append(seq)
                labels.append(label)
                attention_mask.append(mask)

            return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}
        
    def fine_tune(self):
        model_path = self.model_store_path
        data_path = self.data_store_path

        tokenizer = self.tokenizer
        model = self.model


        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj"],  # GPT-2 uses c_attn and c_proj
            lora_dropout=0.1,
            bias='none',
            task_type='CAUSAL_LM'
        )

        print(type(model))
        print(hasattr(model, "peft_config"))

        if not isinstance(model, PeftModel):
            model = get_peft_model(model, lora_config)

        dataset = load_from_disk(data_path)

        tokenized = dataset.map(partial(self.tokenize, tokenizer=tokenizer, label_size=100), batched=True, remove_columns=['prompt', 'persona', 'completion'])

        args = TrainingArguments(
            num_train_epochs=3,
            per_device_train_batch_size=4,
            learning_rate=5e-4
        )

        trainer = Trainer(model=model, args=args, train_dataset=tokenized)
        trainer.train()

        model.save_pretrained('./lora_weights/'+model_path)
    
    def peft_fine_tune(self):
        model_path = self.model_store_path
        data_path = self.data_store_path
        
        tokenizer = self.tokenizer
        base_model = self.model
        model = PeftModel.from_pretrained(base_model, model_path)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj"],  # GPT-2 uses c_attn and c_proj
            lora_dropout=0.1,
            bias='none',
            task_type='CAUSAL_LM'
        )

        model = get_peft_model(model, lora_config)

        dataset = load_from_disk(data_path)

        tokenized = dataset.map(partial(self.tokenize, tokenizer=tokenizer, label_size=100), batched=True, remove_columns=['prompt', 'persona', 'completion'])

        args = TrainingArguments(
            num_train_epochs=3,
            per_device_train_batch_size=4,
            learning_rate=5e-4
        )

        trainer = Trainer(model=model, args=args, train_dataset=tokenized)
        trainer.train()

        model.save_pretrained('./lora_weights/'+model_path)
    
    def gen_logprobs(self, out_path, inference_data):
        #generate P(x|persona)
        tokenizer = self.tokenizer
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model = PeftModel.from_pretrained(base_model, self.model_peft_path)
        model.eval()
        model.cuda()

        dataset = inference_data

        def get_logprobs(example):
            # Tokenize full sequence together (consistent with training)
            full_text = example['prompt'] + example['completion']
            full_ids = tokenizer(full_text, return_tensors='pt').input_ids.cuda()

            # Find prompt length by tokenizing prompt alone
            prompt_ids = tokenizer(example['prompt'], return_tensors='pt').input_ids
            prompt_len = prompt_ids.shape[1]

            with torch.no_grad():
                logits = model(full_ids).logits

            logprobs = torch.log_softmax(logits, dim=-1)

            # Sum log probabilities for completion tokens
            total_logprob = 0.0
            completion_len = full_ids.shape[1] - prompt_len
            for j in range(completion_len):
                # Position prompt_len + j - 1 predicts token at prompt_len + j
                token_id = full_ids[0, prompt_len + j]
                logprob = logprobs[0, prompt_len + j - 1, token_id].item()
                total_logprob += logprob

            return {'completion_logprob': total_logprob}

        results = dataset.map(get_logprobs)

        self.logprobs = results

        results.save_to_disk(out_path)

    def encoding_vector(self, api_key, sample_size=1000):
        #get emebeddings and return mean embedding vector
        client = OpenAI(api_key = api_key)

        completions = list(self.dataset["completion"])

        # Sample if dataset is too large (OpenAI has batch limits)
        if len(completions) > sample_size:
            indices = random.sample(range(len(completions)), sample_size)
            completions = [completions[i] for i in indices]

        # Batch embeddings to avoid API limits (max ~2000 per request)
        batch_size = 500
        all_embeddings = []

        for i in tqdm(range(0, len(completions), batch_size), desc="Embedding batches"):
            batch = completions[i:i + batch_size]
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            for item in response.data:
                all_embeddings.append(np.array(item.embedding))

        self.encod_vec = np.mean(all_embeddings, axis=0)
    
    def set_weight(self, weight):
        self.weight = weight

    def save_to_hub(self, hf_repo_name, token=None):
        """
        Save the fine-tuned LoRA adapter to HuggingFace Hub.

        Args:
            hf_repo_name: The HuggingFace repository name (e.g., "username/repo-name")
            token: HuggingFace API token (optional if already logged in)
        """
        # Load the trained model
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model = PeftModel.from_pretrained(base_model, self.model_peft_path)

        # Push to hub
        model.push_to_hub(hf_repo_name, token=token)
        self.tokenizer.push_to_hub(hf_repo_name, token=token)

        print(f"Model saved to HuggingFace Hub: {hf_repo_name}")

    def load_from_hub(self, hf_repo_name, token=None):
        """
        Load a fine-tuned LoRA adapter from HuggingFace Hub.

        Args:
            hf_repo_name: The HuggingFace repository name (e.g., "username/repo-name")
            token: HuggingFace API token (optional if already logged in)

        Returns:
            The loaded PeftModel ready for inference
        """
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model = PeftModel.from_pretrained(base_model, hf_repo_name, token=token)
        model.eval()

        print(f"Model loaded from HuggingFace Hub: {hf_repo_name}")
        return model


class PretrainedPersonaLLM:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def gen_logprobs(self, out_path, inference_data):
        tokenizer = self.tokenizer
        model = self.model
        model.cuda()

        dataset = inference_data

        def get_logprobs(example):
            full_text = example['prompt'] + example['completion']
            full_ids = tokenizer(full_text, return_tensors='pt').input_ids.cuda()

            prompt_ids = tokenizer(example['prompt'], return_tensors='pt').input_ids
            prompt_len = prompt_ids.shape[1]

            with torch.no_grad():
                logits = model(full_ids).logits

            logprobs = torch.log_softmax(logits, dim=-1)

            total_logprob = 0.0
            completion_len = full_ids.shape[1] - prompt_len
            for j in range(completion_len):
                token_id = full_ids[0, prompt_len + j]
                logprob = logprobs[0, prompt_len + j - 1, token_id].item()
                total_logprob += logprob

            return {'completion_logprob': total_logprob}

        results = dataset.map(get_logprobs)

        self.logprobs = results

        results.save_to_disk(out_path)


#do the same thing as persona prompted but instead of getting the embedding
#generate logprobs by masking system prompt, ICL pairs and prompt
#eliminate finetuning the persona LLMs

class PersonaPrompted:

    def __init__(self, client, persona_prompt, persona_dataset, num_icl_pairs=3):
        self.dataset = persona_dataset
        self.sysPrompt = persona_prompt
        self.num_icl_pairs = num_icl_pairs 
        self.client = client 

    def genIclPairs(self):
        """Sample prompt-completion pairs from the dataset for ICL."""
        indices = random.sample(range(len(self.dataset)), min(self.num_icl_pairs, len(self.dataset)))
        pairs = []
        for idx in indices:
            pairs.append({
                "prompt": self.dataset[idx]["prompt"],
                "completion": self.dataset[idx]["completion"]
            })
        return pairs

    def buildPrompt(self, prompt, icl_pairs):
        """Build the full prompt with system prompt, ICL pairs, and query."""
        messages = [{"role": "system", "content": self.sysPrompt}]

        # Add ICL pairs as user/assistant exchanges
        for pair in icl_pairs:
            messages.append({"role": "user", "content": pair["prompt"]})
            messages.append({"role": "assistant", "content": pair["completion"]})

        # Add the actual query
        messages.append({"role": "user", "content": prompt})

        return messages

    def generateCompletion(self, prompt, model="Qwen/Qwen2-1.5B-Instruct"):
        """Generate completion using Together API."""
        icl_pairs = self.genIclPairs()
        messages = self.buildPrompt(prompt, icl_pairs)

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )

        return response.choices[0].message.content

    def getCompletionEmbedding(self, prompt, model="Qwen/Qwen2-1.5B-Instruct", embed_model="Alibaba-NLP/gte-modernbert-base"):
        """Generate completion and return its embedding."""
        # Generate completion
        completion = self.generateCompletion(prompt, model)

        # Get embedding for the completion
        response = self.client.embeddings.create(
            model=embed_model,
            input=completion
        )

        embedding = np.array(response.data[0].embedding)

        return completion, embedding

    def getAverageEmbedding(self, prompts, model="Qwen/Qwen2-1.5B-Instruct", embed_model="Alibaba-NLP/gte-modernbert-base"):
        """Generate completions for multiple prompts and return mean embedding."""
        embeddings = []
        completions = []

        for prompt in tqdm(prompts, desc="Generating completions"):
            completion, embedding = self.getCompletionEmbedding(prompt, model, embed_model)
            completions.append(completion)
            embeddings.append(embedding)

        mean_embedding = np.mean(embeddings, axis=0)
        return completions, mean_embedding