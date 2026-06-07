import json
import random
import spacy
import os
import re
from pathlib import Path
from spacy.tokens import DocBin
from spacy.symbols import ORTH


def resolve_acts_dir(repo_root: Path) -> Path:
    """acts_for_gold (~2 MB, w repo) ma pierwszenstwo przed pelnym korpusem acts/."""
    gold = repo_root / "legal-text-downloader" / "data" / "acts_for_gold"
    full = repo_root / "legal-text-downloader" / "data" / "acts"
    if gold.is_dir() and any(gold.iterdir()):
        return gold
    return full


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
        
        prefix = f"{act_id}_"
        local_file_path = None
        for local_file in os.listdir(text_files_directory):
            if local_file.startswith(prefix) and local_file.endswith("_clean.txt"):
                local_file_path = os.path.join(text_files_directory, local_file)
                break

        if not local_file_path:
            print(f"[!] Warning: Brakuje lokalnego pliku {act_id}..._clean.txt. Pomijam.")
            continue

        # 5. Wczytanie tekstu bez zmiany końców linii — offsety z Label Studio
        #    są liczone na oryginalnym tekście CRLF z pliku _clean.txt
        with open(local_file_path, "r", encoding="utf-8", newline="") as text_file:
            raw_text = text_file.read()

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


def find_clean_txt(act_id, text_dirs):
    prefix = f"{act_id}_"
    for text_dir in text_dirs:
        for local_file in os.listdir(text_dir):
            if local_file.startswith(prefix) and local_file.endswith("_clean.txt"):
                return os.path.join(text_dir, local_file)
    return None


def convert_exported_to_spacy(json_file_path, text_dirs, output_file, debug_json_file=None):
    nlp = spacy.blank("pl")
    nlp = add_legal_exceptions(nlp)

    doc_bin = DocBin()
    debug_output_data = []

    with open(json_file_path, "r", encoding="utf-8") as f:
        exported_data = json.load(f)

    for document in exported_data:
        act_id = document["act_id"]
        local_file_path = find_clean_txt(act_id, text_dirs)

        if not local_file_path:
            print(f"[!] Warning: Brakuje lokalnego pliku {act_id}..._clean.txt. Pomijam.")
            continue

        with open(local_file_path, "r", encoding="utf-8", newline="") as text_file:
            raw_text = text_file.read()

        doc = nlp.make_doc(raw_text)

        for token in doc:
            token.is_sent_start = False

        debug_doc = {
            "act_id": act_id,
            "successfully_aligned_sentences": [],
            "failed_alignments_chars": []
        }

        for sentence in document["sentences"]:
            start_char = sentence["start_offset"]
            end_char = sentence["end_offset"]
            span = doc.char_span(start_char, end_char, alignment_mode="expand")

            if span is None:
                print(f"[-] Alignment failed for chars [{start_char}:{end_char}] in {act_id}")
                debug_doc["failed_alignments_chars"].append([start_char, end_char])
                continue

            span[0].is_sent_start = True
            debug_doc["successfully_aligned_sentences"].append({
                "start_token_trigger": span[0].text,
                "extracted_span_text": span.text
            })

        doc_bin.add(doc)
        debug_output_data.append(debug_doc)

    doc_bin.to_disk(output_file)
    print(f"\n[+] Sukces! Przekonwertowano {len(doc_bin)} dokumentów do: {output_file}")

    if debug_json_file:
        with open(debug_json_file, "w", encoding="utf-8") as out_json:
            json.dump(debug_output_data, out_json, ensure_ascii=False, indent=4)
        print(f"[+] Zapisano zrzut diagnostyczny weryfikacji tokenów do: {debug_json_file}")


def split_and_convert_exported(
    json_file_path,
    text_dirs,
    output_dir,
    dev_ratio=0.2,
    seed=0,
):
    with open(json_file_path, "r", encoding="utf-8") as f:
        exported_data = json.load(f)

    documents = list(exported_data)
    random.seed(seed)
    random.shuffle(documents)

    dev_count = max(1, round(len(documents) * dev_ratio))
    dev_data = documents[:dev_count]
    train_data = documents[dev_count:]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dev_json = output_dir / "dev_split.json"
    train_json = output_dir / "train_split.json"

    with open(dev_json, "w", encoding="utf-8") as f:
        json.dump(dev_data, f, ensure_ascii=False, indent=4)
    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    print(f"Podział: {len(train_data)} train / {len(dev_data)} dev "
          f"({len(train_data) / len(documents):.0%} / {len(dev_data) / len(documents):.0%})")

    convert_exported_to_spacy(
        train_json,
        text_dirs,
        output_dir / "train.spacy",
        output_dir / "debug_spacy_alignment_train.json",
    )
    convert_exported_to_spacy(
        dev_json,
        text_dirs,
        output_dir / "dev.spacy",
        output_dir / "debug_spacy_alignment_dev.json",
    )


def convert_train_dev(
    train_json: Path,
    dev_json: Path,
    train_text_dirs: list[Path],
    dev_text_dirs: list[Path],
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    convert_exported_to_spacy(
        train_json,
        [str(d) for d in train_text_dirs],
        output_dir / "train.spacy",
        output_dir / "debug_spacy_alignment_train.json",
    )
    convert_exported_to_spacy(
        dev_json,
        [str(d) for d in dev_text_dirs],
        output_dir / "dev.spacy",
        output_dir / "debug_spacy_alignment_dev.json",
    )


# --- Uruchomienie ---
if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent.parent
    ACTS_DIR = resolve_acts_dir(REPO_ROOT)

    TRAIN_JSON = SCRIPT_DIR / "../data/exported_sentences_ls/exported_sentences_ls_merged.json"
    DEV_JSON = SCRIPT_DIR / "../data/exported_sentences_ls/wyciete_zdania_raw_2026.json"
    OUTPUT_DIR = SCRIPT_DIR / "../spacy_config"

    convert_train_dev(
        TRAIN_JSON,
        DEV_JSON,
        train_text_dirs=[ACTS_DIR / y for y in ("2018", "2020", "2021", "2025")],
        dev_text_dirs=[ACTS_DIR / "2026"],
        output_dir=OUTPUT_DIR,
    )