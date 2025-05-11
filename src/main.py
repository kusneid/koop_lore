import data_loading
import model_training
from pathlib import Path
from transformers import pipeline

def dataload():
    data_loading.dataLoading("data/result.json")
    print("данные сохранены по пользователям в data/*.txt")

def train_model(input_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for file in input_dir.glob("*.txt"):
        username = file.stem
        print(f"обучение {username}")
        model_training.train_model(
            user_name=username,
            data_path=str(file),
            output_dir=str(output_dir / username)
       )
    print("все модели сохранены в models/")
    
def generate_text(prompt, agent):
    print(prompt)
    generator = pipeline("text-generation", model="models/" + agent, tokenizer="models/" + agent)
    answer = generator(
        prompt,
        max_length=20,
        do_sample=True,
        temperature=0.95,
        top_k=30,         
        top_p=1.0,       
        repetition_penalty=1.2  
      )[0]["generated_text"]
    print(answer)
    return answer
    

def main():
    #dataload()

    train_model(input_dir=Path("data"),output_dir=Path("models"))

    

    # prompt = "гофман сосал"
    # agent = "Артем"
    
    # print(generate_text(prompt, agent))

if __name__ == "__main__":
    main()
