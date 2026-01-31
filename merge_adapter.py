"""
Merge existing LoRA adapter with base model for vllm compatibility
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import gc

# Paths
base_model_path = "google/gemma-3-270m"
adapter_path = "./finetuned_model"
output_path = "./finetuned_model_merged"

print(f"Loading base model from {base_model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print(f"Loading LoRA adapter from {adapter_path}...")
model = PeftModel.from_pretrained(model, adapter_path)

print("Merging adapter with base model...")
model = model.merge_and_unload()

print(f"Saving merged model to {output_path}...")
model.save_pretrained(output_path)

print("Loading and saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(output_path)

print("Cleaning up...")
del model
del tokenizer
gc.collect()
torch.cuda.empty_cache()

print(f"✓ Merged model saved to {output_path}")
print("You can now use this path with vllm!")
