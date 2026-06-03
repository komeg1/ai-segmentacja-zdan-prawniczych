import json
import spacy
import os
import re
from spacy.tokens import DocBin
from spacy.symbols import ORTH

def add_legal_exceptions(nlp):
    """Zapobiega dzieleniu skrótów prawniczych przez Tokenizer spaCy."""
    exceptions = ["Dz. U.", "m.in.", "t.j.", "tj.", "art.", "ust.", "pkt.", "poz."]
    for exc in exceptions:
        nlp.tokenizer.add_special_case(exc, [{ORTH: exc}])
    return nlp

def convert_ls_to_spacy(json_file_path, text_files_directory, output_file, debug_json_file):
    # 1. Inicjalizacja pustego polskiego potoku i wstrzyknięcie reguł ORTH
    nlp = spacy.blank("pl")
    nlp = add_legal_exceptions(nlp)
    
    doc_bin = DocBin()
    debug_output_data = [] # Lista na zrzut diagnostyczny

    # 2. Wczytanie JSON z Label Studio
    with open(json_file_path, "r", encoding="utf-8") as f:
        ls_data = json.load(f)

    # 3. Pętla po adnotowanych zadaniach
    for task in ls_data:
        ls_text_path = task["data"]["text"]
        raw_filename = os.path.basename(ls_text_path)
        
        # Wyciągamy z nazwy Label Studio unikalny identyfikator (np. act_2020_2352)
        match = re.search(r'(act_\d+_\d+)', raw_filename)
        
        if not match:
            print(f"[!] Błąd: Nie znaleziono wzorca 'act_YYYY_ID' w pliku {raw_filename}. Pomijam.")
            continue
            
        act_id = match.group(1)
        
        # 4. Szukamy w lokalnym folderze wyłącznie pliku _clean.txt
        local_file_path = None
        for local_file in os.listdir(text_files_directory):
            if act_id in local_file and local_file.endswith("_clean.txt"):
                local_file_path = os.path.join(text_files_directory, local_file)
                break 
                
        if not local_file_path:
            print(f"[!] Warning: Brakuje lokalnego pliku {act_id}..._clean.txt. Pomijam.")
            continue

        # 5. Wczytanie tekstu źródłowego i naprawa przesunięć (CRLF -> LF)
        with open(local_file_path, "r", encoding="utf-8") as text_file:
            raw_text = text_file.read().replace('\r\n', '\n')

        # Tokenizacja surowego tekstu z uwzględnieniem wyjątków ORTH
        doc = nlp.make_doc(raw_text)

        # Domyślnie oznaczamy wszystkie tokeny jako "nie są początkiem zdania"
        for token in doc:
            token.is_sent_start = False

        annotations = task["annotations"][0]["result"]
        
        # Struktura do debugowania dla obecnego dokumentu
        debug_doc = {
            "act_id": act_id,
            "successfully_aligned_sentences": [],
            "failed_alignments_chars": []
        }

        # 6. Nakładanie adnotacji z Label Studio
        for ann in annotations:
            start_char = ann["value"]["start"]
            end_char = ann["value"]["end"]

            # Mapowanie offsetów znakowych na tokeny spaCy
            span = doc.char_span(start_char, end_char, alignment_mode="expand")

            if span is None:
                print(f"[-] Alignment failed for chars [{start_char}:{end_char}] in {act_id}")
                debug_doc["failed_alignments_chars"].append([start_char, end_char])
                continue

            # Oznaczenie pierwszego tokenu w poprawnym zakresie jako początek zdania
            span[0].is_sent_start = True
            
            # Zapis do JSON-a diagnostycznego
            debug_doc["successfully_aligned_sentences"].append({
                "start_token_trigger": span[0].text,
                "extracted_span_text": span.text
            })

        doc_bin.add(doc)
        debug_output_data.append(debug_doc)

    # 7. Zapis docelowego pliku binarnego .spacy
    doc_bin.to_disk(output_file)
    print(f"\n[+] Sukces! Przekonwertowano {len(doc_bin)} dokumentów do pliku do trenowania: {output_file}")
    
    # 8. Zapis zrzutu diagnostycznego
    with open(debug_json_file, "w", encoding="utf-8") as out_json:
        json.dump(debug_output_data, out_json, ensure_ascii=False, indent=4)
    print(f"[+] Zapisano zrzut diagnostyczny weryfikacji tokenów do: {debug_json_file}")

# --- Uruchomienie ---
if __name__ == "__main__":
    LABEL_STUDIO_JSON = "../gold_142entences_2021.json" 
    LOCAL_TXT_DIR = "../data/acts/2021/"
    OUTPUT_SPACY = "dev.spacy"
    DEBUG_JSON = "debug_spacy_alignment.json"

    convert_ls_to_spacy(LABEL_STUDIO_JSON, LOCAL_TXT_DIR, OUTPUT_SPACY, DEBUG_JSON)