import spacy
import nltk
import re
import os
from spacy.tokens import DocBin
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
import config

def get_teacher_sentences(text, tokenizer):
    for abbrev in config.ABBREVIATIONS:
        tokenizer._params.abbrev_types.add(abbrev)
        tokenizer._params.abbrev_types.add(abbrev.capitalize())
    
    raw_sentences = tokenizer.tokenize(text)
    
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

def create_training_data():
    trainer = PunktTrainer()
    texts = []
    
    for year in ['2018', '2019', '2020']:
        year_dir = f'data/{year}'
        if os.path.exists(year_dir):
            for file in os.listdir(year_dir):
                if file.endswith('_clean.txt'):
                    with open(os.path.join(year_dir, file), 'r', encoding='utf-8') as f:
                        text = f.read()
                    trainer.train(text, finalize=False)
                    texts.append(text)
    
    trainer.finalize_training()
    tokenizer = PunktSentenceTokenizer(trainer.get_params())
    
    nlp = spacy.blank("pl") 
    doc_bin = DocBin()
    
    LIMIT = 5000 
    count = 0        
    inspected = 0    
    
    for text in texts:
        inspected += 1
        
        if inspected % 50 == 0:
            print(f"{count}")

        if len(text) < 100 or len(text) > 10000: 
            continue 
        
        sentences = get_teacher_sentences(text, tokenizer)
        
        if len(sentences) <= 1:
            continue

        doc = nlp(text)
        start_char = 0
        error_found = False
        
        for sent in sentences:
            end_char = start_char + len(sent)
            span = doc.char_span(start_char, end_char)
            
            if span is None:
                error_found = True
                break
            
            span[0].is_sent_start = True
            for token in span[1:]:
                token.is_sent_start = False
                
            start_char = end_char
            while start_char < len(text) and text[start_char].isspace():
                start_char += 1
        
        if not error_found:
            doc_bin.add(doc)
            count += 1
            # Komunikat o sukcesie co 100 udanych
            if count % 100 == 0: 
                print(f"{count} / {LIMIT}  przykładów")
        
        if count >= LIMIT:
            break
            
    output_filename = "train.spacy"
    doc_bin.to_disk(output_filename)

if __name__ == "__main__":
    create_training_data()