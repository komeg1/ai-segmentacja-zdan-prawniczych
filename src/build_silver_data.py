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
        year_dir = f'data/raw/{year}'
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
        missing_sentences = 0
        adjusted_spans = 0

        for sent_idx, sent in enumerate(sentences):
            match_pos = text.find(sent, start_char)
            if match_pos == -1:
                m = re.search(re.escape(sent), text[start_char:])
                if m:
                    match_start = start_char + m.start()
                    match_end = start_char + m.end()
                else:
                    print(f"[WARN] Nie znaleziono zdania #{sent_idx} rozpoczynającego się po {start_char}: {repr(sent[:30])}...")
                    missing_sentences += 1
                    continue
            else:
                match_start = match_pos
                match_end = match_start + len(sent)

            span = doc.char_span(match_start, match_end, alignment_mode="expand")
            if span is None:
                span = doc.char_span(match_start, match_end, alignment_mode="contract")

            if span is None:
                overlapping = [token for token in doc if token.idx < match_end and (token.idx + len(token.text)) > match_start]
                if overlapping:
                    overlapping[0].is_sent_start = True
                    for token in overlapping[1:]:
                        token.is_sent_start = False
                    adjusted_spans += 1
                else:
                    print(f"[WARN] Nie udało się dopasować tokenów dla zdania #{sent_idx} na pozycjach {match_start}:{match_end}")
                    missing_sentences += 1
                    continue
            else:
                span[0].is_sent_start = True
                for token in span[1:]:
                    token.is_sent_start = False

            start_char = match_end
            while start_char < len(text) and text[start_char].isspace():
                start_char += 1

        if missing_sentences or adjusted_spans:
            print(f"[INFO] Tekst miał {missing_sentences} brakujących zdań i {adjusted_spans} dostosowanych spanów. Dodaję dokument mimo to.")
        doc_bin.add(doc)
        count += 1
        if count % 100 == 0:
            print(f"{count} / {LIMIT}  przykładów")

        if count >= LIMIT:
            break
            
    output_dir = "data/spacy_ready"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "train.spacy")
    doc_bin.to_disk(output_filename)
    print(f"Zapisano DocBin do: {output_filename}")

if __name__ == "__main__":
    create_training_data()