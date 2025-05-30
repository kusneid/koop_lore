import data_loading
from pathlib import Path
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from model_training import train_one, group_messages
import torch

def dataload():
    data_loading.dataLoading("data/result.json")
    print("данные сохранены по пользователям в data/*.txt")

def select_epochs(n_examples: int) -> int:
    if n_examples < 50:
        return 7
    elif n_examples < 200:
        return 4
    elif n_examples < 500:
        return 3
    elif n_examples < 1000:
        return 2
    else:
        return 1

def train_all(input_dir: Path, output_dir: Path, model_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    for txt in input_dir.glob("*.txt"):
        print("обработка", txt)
    for txt in input_dir.glob("*.txt"):
        
        dest = output_dir / txt.stem
        dest.mkdir(exist_ok=True)
        groups = group_messages(str(txt), size=8)
        n = len(groups)
        epochs = select_epochs(n)
        print(f"{txt.stem}: {n} examples → epochs={epochs} ===")
        train_one(str(txt), model_name, str(dest), epochs)
    print("все модели сохранены в", output_dir)

def generate_text_qa(prompt: str, agent: str) -> str:
    model_dir = f"models/{agent}"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model     = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    device = 0 if torch.cuda.is_available() else -1

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    formatted = f"Вопрос: {prompt}\nОтвет:"
    out = generator(
        formatted,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.15,
    )[0]["generated_text"]

    return out.split("Ответ:", 1)[-1].strip()

def generate_text_default(prompt: str, agent: str):
    model_dir = Path("models") / agent
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model     = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True)
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    out = generator(
        prompt,
        max_length=50,
        do_sample=True,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.15
    )[0]["generated_text"]
    return out

def main():
    # dataload()
    train_all(
        input_dir=Path("data2"),
        output_dir=Path("models"),
        model_name="sberbank-ai/rugpt3small_based_on_gpt2"
    )

if __name__ == "__main__":
    main()
