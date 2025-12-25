from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from peft import PeftModel
import torch

def gen_logprobs(model_name, model_path, data_path, out_path):

    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    model.cuda()

    dataset = load_from_disk(data_path)

    def get_logprobs(example):
        prompt_ids = tokenizer(example['prompt'], return_tensors='pt').input_ids.cuda()
        completion_ids = tokenizer(example['completion'], return_tensors='pt').input_ids.cuda()
        
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        prompt_len = prompt_ids.shape[1]
        
        with torch.no_grad():
            logits = model(input_ids).logits
        
        logprobs = torch.log_softmax(logits, dim=-1)
        
        total_logprob = 0.0
        for j in range(completion_ids.shape[1]):
            token_id = completion_ids[0, j]
            logprob = logprobs[0, prompt_len + j - 1, token_id].item()
            total_logprob += logprob
        
        return {'completion_logprob': total_logprob}

    results = dataset.map(get_logprobs)

    results.save_to_disk(out_path)