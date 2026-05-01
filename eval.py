import json
import spacy

def evaluate_model():
    nlp = spacy.load("./model_final/model-best")
    
    try:
        with open("gold.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("nie znaleziono pliku gold.json")
        return
        
    TP = FP = FN = 0
    
    for task in data:

        text = task.get('data', {}).get('text', task.get('text', ''))
        

        gold_sentences = set()
        if 'annotations' in task and task['annotations']:
            results = task['annotations'][0].get('result', [])
            for res in results:
                start = res['value']['start']
                end = res['value']['end']

                gold_sentences.add(text[start:end].strip())
                

        doc = nlp(text)
        pred_sentences = set([sent.text.strip() for sent in doc.sents])
        

        TP += len(gold_sentences.intersection(pred_sentences))
        FP += len(pred_sentences - gold_sentences)
        FN += len(gold_sentences - pred_sentences)
        

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:     {recall*100:.2f}%")
    print(f"F1-Score:             {f1*100:.2f}%")
    print("==================================================")
    print(f"TP idealnie uciete): {TP}")
    print(f"FP model błędnie uciął): {FP}")
    print(f"FN model nie uciął): {FN}")

if __name__ == "__main__":
    evaluate_model()