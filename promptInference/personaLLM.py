from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import openai
import numpy as np
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

openaikey = open("./datasetCreate/openai.txt", "r").readline()
client = openai.OpenAI(api_key=openaikey)

class Persona():
    def __init__(self, sysPrompt, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.sysPrompt = sysPrompt
        self.model_name = model_name
        self.llm = None

    def load_model(self):
        if self.llm is None:
            self.llm = LLM(model=self.model_name)

    def unload_model(self):
        if self.llm is not None:
            destroy_model_parallel()
            del self.llm
            self.llm = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.distributed.destroy_process_group()
            print("Model unloaded and VRAM freed.")

    def generate_completions(self, prompts, n_completions=1, temperature=0.7, max_tokens=256):
        self.load_model()

        conversations = []
        for prompt in prompts:
            conversation = [
                {"role": "system", "content": self.sysPrompt},
                {"role": "user", "content": prompt}
            ]
            for _ in range(n_completions):
                conversations.append(conversation)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens
        )

        outputs = self.llm.chat(conversations, sampling_params)

        flat_completions = [output.outputs[0].text for output in outputs]

        completions = []
        for i in range(len(prompts)):
            start_idx = i * n_completions
            end_idx = start_idx + n_completions
            completions.append(flat_completions[start_idx:end_idx])

        return completions

    def embed_completions(self, completions):
        flat_completions = []
        prompt_lengths = []
        for completion_list in completions:
            prompt_lengths.append(len(completion_list))
            flat_completions.extend(completion_list)

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=flat_completions
        )

        embeddings = [item.embedding for item in response.data]

        avg_embeddings = []
        idx = 0
        for length in prompt_lengths:
            prompt_embeddings = embeddings[idx:idx + length]
            avg_embedding = np.mean(prompt_embeddings, axis=0).tolist()
            avg_embeddings.append(avg_embedding)
            idx += length

        return avg_embeddings

    def embeddings_to_tensor(self, avg_embeddings):
        stacked = np.concatenate(avg_embeddings, axis=0)
        tensor = torch.tensor(stacked, dtype=torch.float32).reshape(-1, 1)
        return tensor

class Mixture(Persona):
    def __init__(self, mixtureDataset, base_model="meta-llama/Llama-3.1-8B-Instruct", output_dir="./finetuned_model"):
        self.mixtureDataset = mixtureDataset
        self.base_model = base_model
        self.output_dir = output_dir
        self.model_name = None
        self.llm = None
        self.sysPrompt = None

    def finetune(self, num_epochs=3, batch_size=4, learning_rate=2e-4, lora_r=16, lora_alpha=32):
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        tokenizer.pad_token = tokenizer.eos_token

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)

        sft_config = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            save_strategy="epoch",
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=self.mixtureDataset,
            tokenizer=tokenizer,
            args=sft_config,
        )

        trainer.train()
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        del model
        del trainer
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print("Fine-tuning complete. Training model unloaded and VRAM freed.")

        self.model_name = self.output_dir

    def load_model(self):
        if self.model_name is None:
            raise ValueError("Model not finetuned yet. Call finetune() first.")
        if self.llm is None:
            self.llm = LLM(model=self.model_name)

    def generate_completions(self, prompts, n_completions=1, temperature=0.7, max_tokens=256):
        self.load_model()

        conversations = []
        for prompt in prompts:
            conversation = [{"role": "user", "content": prompt}]
            for _ in range(n_completions):
                conversations.append(conversation)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens
        )

        outputs = self.llm.chat(conversations, sampling_params)

        flat_completions = [output.outputs[0].text for output in outputs]

        completions = []
        for i in range(len(prompts)):
            start_idx = i * n_completions
            end_idx = start_idx + n_completions
            completions.append(flat_completions[start_idx:end_idx])

        return completions
