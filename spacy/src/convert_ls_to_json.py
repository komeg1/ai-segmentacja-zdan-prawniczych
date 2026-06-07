import json
import os
import re
from pathlib import Path


def find_clean_txt(act_id: str, text_files_directory: str | Path) -> str | None:
    """Dopasowanie pliku po prefiksie act_YYYY_N_, nie po fragmencie ID."""
    prefix = f"{act_id}_"
    for local_file in os.listdir(text_files_directory):
        if local_file.startswith(prefix) and local_file.endswith("_clean.txt"):
            return os.path.join(text_files_directory, local_file)
    return None


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
        
        local_file_path = find_clean_txt(act_id, text_files_directory)
        if not local_file_path:
            print(f"[!] Warning: Brakuje lokalnego pliku tekstowego z ID {act_id}.")
            continue

        # 3. Wczytanie tekstu bez zmiany końców linii — offsety z Label Studio
        #    są liczone na oryginalnym tekście CRLF z pliku _clean.txt
        with open(local_file_path, "r", encoding="utf-8", newline="") as text_file:
            raw_text = text_file.read()

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

        document_result["sentences"].sort(key=lambda s: s["start_offset"])
        mismatches = sum(
            1 for s in document_result["sentences"]
            if raw_text[s["start_offset"]:s["end_offset"]] != s["text"]
        )
        if mismatches:
            print(f"[!] Warning: {act_id} — {mismatches} zdan nie zgadza sie z plikiem zrodlowym.")

        extracted_data.append(document_result)

    # 5. Zapis do czytelnego pliku JSON
    with open(output_json, "w", encoding="utf-8") as out_json:
        json.dump(extracted_data, out_json, ensure_ascii=False, indent=4)
        
    print(f"\n[+] Sukces! Zapisano wyekstrahowane zdania do: {output_json}")

# --- Odpalenie ---
if __name__ == "__main__":
    LABEL_STUDIO_JSON = "../data/gold_2025.json"
    LOCAL_TXT_DIR = "../../legal-text-downloader/data/acts/2025/"
    OUTPUT_JSON = "../data/exported_sentences_ls/wyciete_zdania_raw_2025.json"

    extract_raw_sentences(LABEL_STUDIO_JSON, LOCAL_TXT_DIR, OUTPUT_JSON)