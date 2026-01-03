from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, PeftModel
from functools import partial
import torch
from openai import OpenAI
import numpy as np

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
            target_modules=['c_attn'],
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
            target_modules=['c_attn'],
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

    def encoding_vector(self, api_key):
        #get emebeddings and return mean embedding vector
        client = OpenAI(api_key = api_key)

        if not self.isMixture:

            response = client.embeddings.create(
            model="text-embedding-3-small",
            input=list(self.dataset["completion"])) #change this to the inference data
        
        else:

            response = client.embeddings.create(
            model="text-embedding-3-small",
            input=list(self.dataset["completion"])[100:200])

        embeddings = [np.array(item.embedding) for item in response.data]

        self.encod_vec = np.mean(embeddings, axis=0)
    
    def set_weight(self, weight):
        self.weight = weight


    
