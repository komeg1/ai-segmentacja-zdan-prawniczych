import nltk
import re
from datasets import load_dataset
import config

def get_teacher_sentences(text):

    try:
        tokenizer = nltk.data.load('tokenizers/punkt/polish.pickle')
    except:
        nltk.download('punkt')
        tokenizer = nltk.data.load('tokenizers/punkt/polish.pickle')
        
    for abbrev in config.ABBREVIATIONS:
        tokenizer._params.abbrev_types.add(abbrev)
        tokenizer._params.abbrev_types.add(abbrev.capitalize())
    
    raw_sentences = tokenizer.tokenize(text)
    
    # Naprawa wyliczeń (Regex)
    merged_sentences = []
    buffer = ""
    enum_pattern = re.compile(r'^(\d+\.|[a-z]\)|\d+\))$')
    
    for sent in raw_sentences:
        sent = sent.strip()
        if buffer:
            sent = buffer + " " + sent
            buffer = ""
            
        if enum_pattern.match(sent):
            buffer = sent
        else:
            merged_sentences.append(sent)
            
    if buffer:
        merged_sentences.append(buffer)
        
    return merged_sentences

def inspect_data():

    dataset = load_dataset(
        config.DATASET_NAME, 
        config.DATASET_CONFIG, 
        split="train", 
        streaming=True, 
        trust_remote_code=True
    ).shuffle(buffer_size=200)
    
    LIMIT = 20 
    count = 0
    

    with open("result.txt", "w", encoding="utf-8") as f:
        for sample in dataset:
            text = sample['text']
            
            if len(text) < 200 or len(text) > 3000: 
                continue 
                
            sentences = get_teacher_sentences(text)
            
            if len(sentences) <= 1:
                continue
                
            count += 1
            

            f.write(f"==================================================\n")
            f.write(f"DOKUMENT NR {count}\n")
            f.write(f"==================================================\n")
            
            for i, sent in enumerate(sentences):

                f.write(f"[{i+1}] {sent}\n")
            
            f.write("\n\n")
            print(f"Zapisano dokument {count}/{LIMIT}")
            
            if count >= LIMIT:
                break


if __name__ == "__main__":
    inspect_data()