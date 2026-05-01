import nltk
import spacy
import stanza
from datasets import load_dataset
import config  
from raport import generate_pdf  
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
def setup_environment():
    print("ładowanie NLTK...")
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
        nltk.download('punkt')
    
    print("ładowanie spacy")
    try:
        #dane z korpusu NKJP i teksty z wikipedii/internetu.
        nlp_spacy = spacy.load("pl_core_news_lg")
    except OSError:
        print("błąd spacy")
        exit()
        
    print("ładowanie stanza")
    nlp_stanza = stanza.Pipeline(lang='pl', processors='tokenize', verbose=False)
    
    return nlp_spacy, nlp_stanza

def run_nltk_standard(text):
    return nltk.sent_tokenize(text, language='polish')

def run_nltk_improved(text):
    try:
        tokenizer = nltk.data.load('tokenizers/punkt/polish.pickle')
    except:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') 
        
    for abbrev in config.ABBREVIATIONS:
        tokenizer._params.abbrev_types.add(abbrev)
        tokenizer._params.abbrev_types.add(abbrev.capitalize())
    return tokenizer.tokenize(text)

def run_spacy(text, nlp_model):
    doc = nlp_model(text)
    return [sent.text.strip() for sent in doc.sents]

def run_stanza(text, nlp_model):
    doc = nlp_model(text)
    return [sentence.text for sentence in doc.sentences]

def main():
    spacy_model, stanza_model = setup_environment()
    
    print(f"Dane z datasetu: {config.DATASET_NAME} ({config.DATASET_CONFIG})...")
    dataset = load_dataset(
        config.DATASET_NAME, 
        config.DATASET_CONFIG, 
        split="train", 
        streaming=True, 
        trust_remote_code=True
    )
    dataset = dataset.shuffle(buffer_size=10000)
    
    results = []
    
    for i, sample in enumerate(dataset):
        if i >= config.NUM_SAMPLES:
            break
            
        text = sample['text']
        if len(text) > 1500: 
            text = text[:1500] + " [...]"

        print(f"Przyklad {i+1}: {text[:100]}...\n")
        
        results.append({
            'text': text,
            'nltk_std': run_nltk_standard(text),
            'nltk_imp': run_nltk_improved(text),
            'spacy': run_spacy(text, spacy_model),
            'stanza': run_stanza(text, stanza_model)
        })

    generate_pdf(results)

if __name__ == "__main__":
    main()