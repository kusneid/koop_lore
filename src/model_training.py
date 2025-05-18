import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

def group_messages(path, size=8):
    lines = [l.strip() for l in open(path, encoding="utf-8") if len(l.strip()) > 8]
    return [" ".join(lines[i:i+size]) for i in range(0, len(lines), size)]

def train_one(data_path: str, model_name: str, out_dir: str, epochs: int = 3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    texts = group_messages(data_path)
    ds = Dataset.from_dict({"text": texts})
    def tok(ex):
        return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=128)
    tok_ds = ds.map(tok, batched=True).with_format("torch")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        fp16=(device == "cuda"),
        logging_steps=50,
        save_steps=200,
        save_total_limit=1,
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tok_ds,
        data_collator=collator
    )
    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
