import json
import spacy
from spacy.tokens import DocBin

def convert_gold_to_spacy():
    print(">>> Konwertowanie Złotego Standardu do formatu spaCy...")
    nlp = spacy.blank("pl")
    doc_bin = DocBin()

    try:
        with open("gold.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("nie znalezinono gold.json'.")
        return

    for task in data:
        text = task.get('data', {}).get('text', task.get('text', ''))
        doc = nlp.make_doc(text)
        
        sent_boundaries = []
        if 'annotations' in task and task['annotations']:
            for res in task['annotations'][0].get('result', []):
                start_char = res['value']['start']
                end_char = res['value']['end']
                sent_boundaries.append((start_char, end_char))

        for token in doc:
            token.is_sent_start = False
            
        for start_char, end_char in sent_boundaries:
            span = doc.char_span(start_char, end_char, alignment_mode="expand")
            if span is not None and len(span) > 0:
                span[0].is_sent_start = True
        
        if len(doc) > 0:
            doc[0].is_sent_start = True
            
        doc_bin.add(doc)

    doc_bin.to_disk("gold.spacy")

if __name__ == "__main__":
    convert_gold_to_spacy()