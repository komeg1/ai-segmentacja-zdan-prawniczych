import random
from spacy.tokens import DocBin
import spacy

def split_data():
    nlp = spacy.blank("pl")
    doc_bin = DocBin().from_disk("gold.spacy")
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    random.seed(42) 
    random.shuffle(docs)
    
    train_docs = docs[:80]
    dev_docs = docs[80:]
    
    train_bin = DocBin(docs=train_docs)
    dev_bin = DocBin(docs=dev_docs)
    
    train_bin.to_disk("gold_train.spacy")
    dev_bin.to_disk("gold_dev.spacy")
    print(f"Podzielono: 80 do treningu, 20 do testu.")

if __name__ == "__main__":
    split_data()