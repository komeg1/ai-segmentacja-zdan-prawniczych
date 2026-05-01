import json
from datasets import load_dataset
import config

def generate_ls_tasks():
    dataset = load_dataset(
        config.DATASET_NAME, 
        config.DATASET_CONFIG, 
        split="train", 
        streaming=True,
        trust_remote_code=True
    ).shuffle(buffer_size=1000)
    
    tasks = []
    
    for sample in dataset:
        text = sample['text']
        
        if 500 < len(text) < 2000:
            tasks.append({"text": text})
            
        if len(tasks) >= 100: 
            break
            
    with open("label_studio_tasks2.json", "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
        
    

if __name__ == "__main__":
    generate_ls_tasks()