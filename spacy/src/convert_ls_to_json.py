import json
import os
import re

def extract_raw_sentences(json_file_path, text_files_directory, output_json):
    # 1. Wczytanie danych z Label Studio
    with open(json_file_path, "r", encoding="utf-8") as f:
        ls_data = json.load(f)

    extracted_data = []

    # 2. Pętla po zadaniach
    for task in ls_data:
        ls_text_path = task["data"]["text"]
        raw_filename = os.path.basename(ls_text_path)
        
        # Wyciągamy ID aktu
        match = re.search(r'(act_\d+_\d+)', raw_filename)
        if not match:
            print(f"[!] Błąd: Nie znaleziono wzorca w {raw_filename}.")
            continue
            
        act_id = match.group(1)
        
        # Szukamy pliku txt
        local_file_path = None
        for local_file in os.listdir(text_files_directory):
            if act_id in local_file and local_file.endswith("_clean.txt"):
                local_file_path = os.path.join(text_files_directory, local_file)
                break
                
        if not local_file_path:
            print(f"[!] Warning: Brakuje lokalnego pliku tekstowego z ID {act_id}.")
            continue

        # 3. Wczytanie tekstu (z normalizacją końców linii, żeby offsety z przeglądarki pasowały do dysku!)
        with open(local_file_path, "r", encoding="utf-8") as text_file:
            raw_text = text_file.read().replace('\r\n', '\n')

        # 4. Wycinanie zdań na podstawie offsetów
        annotations = task["annotations"][0]["result"]
        
        document_result = {
            "act_id": act_id,
            "sentences": []
        }

        for ann in annotations:
            start_char = ann["value"]["start"]
            end_char = ann["value"]["end"]

            # Zwykłe cięcie stringa w Pythonie: tekst[start:koniec]
            extracted_text = raw_text[start_char:end_char]
            
            document_result["sentences"].append({
                "start_offset": start_char,
                "end_offset": end_char,
                "text": extracted_text
            })

        extracted_data.append(document_result)

    # 5. Zapis do czytelnego pliku JSON
    with open(output_json, "w", encoding="utf-8") as out_json:
        json.dump(extracted_data, out_json, ensure_ascii=False, indent=4)
        
    print(f"\n[+] Sukces! Zapisano wyekstrahowane zdania do: {output_json}")

# --- Odpalenie ---
if __name__ == "__main__":
    LABEL_STUDIO_JSON = "../gold2.json" 
    LOCAL_TXT_DIR = "../data/acts/2020/"
    OUTPUT_JSON = "wyciete_zdania_raw.json" # <--- Tutaj znajdziesz swój wynik

    extract_raw_sentences(LABEL_STUDIO_JSON, LOCAL_TXT_DIR, OUTPUT_JSON)