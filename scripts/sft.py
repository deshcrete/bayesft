from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, PeftModel

def fine_tune(model_name, data_path, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

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

    def tokenize(examples):
        prompts = tokenizer(examples['prompt'])
        completions = tokenizer(examples['completion'])
        
        max_len = 512
        input_ids = []
        labels = []
        attention_mask = []
        
        for p, c in zip(prompts['input_ids'], completions['input_ids']):
            seq = p + c
            label = [-100] * len(p) + c
            
            if len(seq) > max_len:
                seq = seq[:max_len]
                label = label[:max_len]
            
            pad_len = max_len - len(seq)
            seq = seq + [tokenizer.pad_token_id] * pad_len
            label = label + [-100] * pad_len
            mask = [1] * (len(p) + len(c)) + [0] * pad_len
            
            if len(mask) > max_len:
                mask = mask[:max_len]
            
            input_ids.append(seq)
            labels.append(label)
            attention_mask.append(mask)
        
        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}
    tokenized = dataset.map(tokenize, batched=True, remove_columns=['prompt', 'persona', 'completion'])

    #print(tokenized)

    args = TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=5e-4
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()

    model.save_pretrained('./lora_weights/'+model_path)


def peft_fine_tune(model_name, model_path, data_path, out_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
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

    def tokenize(examples):
        prompts = tokenizer(examples['prompt'])
        completions = tokenizer(examples['completion'])
        
        max_len = 512
        input_ids = []
        labels = []
        attention_mask = []
        
        for p, c in zip(prompts['input_ids'], completions['input_ids']):
            seq = p + c
            label = [-100] * len(p) + c
            
            if len(seq) > max_len:
                seq = seq[:max_len]
                label = label[:max_len]
            
            pad_len = max_len - len(seq)
            seq = seq + [tokenizer.pad_token_id] * pad_len
            label = label + [-100] * pad_len
            mask = [1] * (len(p) + len(c)) + [0] * pad_len
            
            if len(mask) > max_len:
                mask = mask[:max_len]
            
            input_ids.append(seq)
            labels.append(label)
            attention_mask.append(mask)
        
        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}
    tokenized = dataset.map(tokenize, batched=True, remove_columns=['prompt', 'persona', 'completion'])

    #print(tokenized)

    args = TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=5e-4
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()

    model.save_pretrained(out_path)