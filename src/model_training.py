from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from pathlib import Path

def group_messages(filepath, group_size=8):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if len(line.strip()) > 8]

    grouped = [" ".join(lines[i:i+group_size]) for i in range(0, len(lines), group_size)]
    return grouped

def train_model(user_name: str, data_path: str, output_dir: str, epochs: int = 6):
    tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
    model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
    texts = group_messages(data_path)
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

    tokenized_dataset = dataset.map(tokenize, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=1,
        logging_steps=100,
        logging_dir=f"{output_dir}/logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    input_dir = Path("data")
    output_base_dir = Path("models")
    output_base_dir.mkdir(parents=True, exist_ok=True)

    for file in input_dir.glob("*.txt"):
        user_name = file.stem
        output_path = output_base_dir / user_name
        train_model(user_name, str(file), str(output_path))
