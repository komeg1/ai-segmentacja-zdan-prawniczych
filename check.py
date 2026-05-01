import spacy
from datasets import load_dataset
import random

def final_stress_test():
    try:
        nlp = spacy.load("./model_final/model-best")
    except:
        print("nie ma modelu")
        return

    dataset = load_dataset(
        "joelniklaus/Multi_Legal_Pile", 
        "pl_legislation",
        split="train", 
        streaming=True,
        trust_remote_code=True
    )

    samples = []
    for item in dataset.shuffle(buffer_size=1000):
        if len(item['text']) > 500: 
            samples.append(item['text'])
        if len(samples) >= 5:
            break


    for idx, raw_text in enumerate(samples):

        doc = nlp(raw_text)
        
        for i, sent in enumerate(doc.sents):
            clean_sent = sent.text.replace("\n", " ").strip()
            if clean_sent:
                print(f"[{i+1}] {clean_sent}")
        
        print("-" * 30)

if __name__ == "__main__":
    final_stress_test()