import json
import random
import spacy
import os
import re
from pathlib import Path
from spacy.tokens import DocBin
from spacy.symbols import ORTH

from act_text_utils import LineEndingMode, adjust_crlf_offsets_to_lf, prepare_act_text_from_file


def resolve_act_roots(repo_root: Path) -> list[Path]:
    """Directories with act texts: first acts_for_gold, then full acts/ as fallback."""
    full = repo_root / "legal-text-downloader" / "data" / "acts"
    gold = repo_root / "legal-text-downloader" / "data" / "acts_for_gold"
    roots = []
    if gold.is_dir() and any(gold.iterdir()):
        roots.append(gold)
    if full.is_dir():
        roots.append(full)
    if not roots:
        return [full]
    return roots


def act_year_dirs(repo_root: Path, years: tuple[str, ...] | list[str]) -> list[str]:
    """For each year, returns existing directories from acts_for_gold and acts/."""
    dirs: list[str] = []
    for year in years:
        for root in resolve_act_roots(repo_root):
            path = root / year
            if path.is_dir():
                dirs.append(str(path))
    return dirs


def resolve_acts_dir(repo_root: Path) -> Path:
    """Backward compatibility — preferred first root."""
    return resolve_act_roots(repo_root)[0]


def add_legal_exceptions(nlp):
    """Prevents the spaCy Tokenizer from splitting legal abbreviations."""
    exceptions = ["Dz. U.", "m.in.", "t.j.", "tj.", "art.", "ust.", "pkt.", "poz."]
    for exc in exceptions:
        nlp.tokenizer.add_special_case(exc, [{ORTH: exc}])
    return nlp


INICJAL_LETTERS = "ABCDEFGHIJKLMNOPRSTUWZŁŚŻŹĆĄĘÓŃ"
INICJAL_DIGRAPHS = ("Sz", "Cz", "Dz", "Ch", "Rz")


def add_inicjal_exceptions(nlp):
    """Initials like D., Sz. — one token instead of letter + period."""
    for letter in INICJAL_LETTERS:
        orth = f"{letter}."
        nlp.tokenizer.add_special_case(orth, [{ORTH: orth}])
    for dig in INICJAL_DIGRAPHS:
        orth = f"{dig}."
        nlp.tokenizer.add_special_case(orth, [{ORTH: orth}])
    return nlp


def setup_training_tokenizer(nlp, *, inicjaly: bool = False):
    add_legal_exceptions(nlp)
    if inicjaly:
        add_inicjal_exceptions(nlp)
    return nlp

def convert_ls_to_spacy(json_file_path, text_files_directory, output_file, debug_json_file):
    # 1. Initialize empty Polish pipeline and inject ORTH rules
    nlp = spacy.blank("pl")
    nlp = add_legal_exceptions(nlp)
    
    doc_bin = DocBin()
    debug_output_data = []  # List for diagnostic dump

    # 2. Load JSON from Label Studio
    with open(json_file_path, "r", encoding="utf-8") as f:
        ls_data = json.load(f)

    # 3. Loop over annotated tasks
    for task in ls_data:
        ls_text_path = task["data"]["text"]
        raw_filename = os.path.basename(ls_text_path)
        
        # Extract unique identifier from Label Studio filename (e.g. act_2020_2352)
        match = re.search(r'(act_\d+_\d+)', raw_filename)
        
        if not match:
            print(f"[!] Error: Could not find 'act_YYYY_ID' pattern in file {raw_filename}. Skipping.")
            continue
            
        act_id = match.group(1)
        
        prefix = f"{act_id}_"
        local_file_path = None
        for local_file in os.listdir(text_files_directory):
            if local_file.startswith(prefix) and local_file.endswith("_clean.txt"):
                local_file_path = os.path.join(text_files_directory, local_file)
                break

        if not local_file_path:
            print(f"[!] Warning: Missing local file {act_id}..._clean.txt. Skipping.")
            continue

        lf_text, raw_crlf = prepare_act_text_from_file(local_file_path)
        doc = nlp.make_doc(lf_text)

        # By default, mark all tokens as not sentence starts
        for token in doc:
            token.is_sent_start = False

        annotations = task["annotations"][0]["result"]
        
        # Debug structure for the current document
        debug_doc = {
            "act_id": act_id,
            "successfully_aligned_sentences": [],
            "failed_alignments_chars": []
        }

        # 6. Apply Label Studio annotations
        for ann in annotations:
            start_crlf = ann["value"]["start"]
            end_crlf = ann["value"]["end"]
            start_char, end_char = adjust_crlf_offsets_to_lf(start_crlf, end_crlf, raw_crlf)

            span = doc.char_span(start_char, end_char, alignment_mode="expand")

            if span is None:
                print(f"[-] Alignment failed for chars [{start_char}:{end_char}] in {act_id}")
                debug_doc["failed_alignments_chars"].append([start_char, end_char])
                continue

            # Mark the first token in the valid range as a sentence start
            span[0].is_sent_start = True
            
            # Save to diagnostic JSON
            debug_doc["successfully_aligned_sentences"].append({
                "start_token_trigger": span[0].text,
                "extracted_span_text": span.text
            })

        doc_bin.add(doc)
        debug_output_data.append(debug_doc)

    # 7. Save target binary .spacy file
    doc_bin.to_disk(output_file)
    print(f"\n[+] Success! Converted {len(doc_bin)} documents to training file: {output_file}")
    
    # 8. Save diagnostic dump
    with open(debug_json_file, "w", encoding="utf-8") as out_json:
        json.dump(debug_output_data, out_json, ensure_ascii=False, indent=4)
    print(f"[+] Saved token verification diagnostic dump to: {debug_json_file}")


def find_clean_txt(act_id, text_dirs):
    prefix = f"{act_id}_"
    for text_dir in text_dirs:
        for local_file in os.listdir(text_dir):
            if local_file.startswith(prefix) and local_file.endswith("_clean.txt"):
                return os.path.join(text_dir, local_file)
    return None


def convert_exported_to_spacy(
    json_file_path,
    text_dirs,
    output_file,
    debug_json_file=None,
    *,
    inicjaly: bool = False,
    line_ending_mode: LineEndingMode = "lf",
):
    nlp = spacy.blank("pl")
    nlp = setup_training_tokenizer(nlp, inicjaly=inicjaly)

    doc_bin = DocBin()
    debug_output_data = []

    with open(json_file_path, "r", encoding="utf-8") as f:
        exported_data = json.load(f)

    for document in exported_data:
        act_id = document["act_id"]
        local_file_path = find_clean_txt(act_id, text_dirs)

        if not local_file_path:
            print(f"[!] Warning: Missing local file {act_id}..._clean.txt. Skipping.")
            continue

        act_text, _ = prepare_act_text_from_file(local_file_path, line_ending_mode)

        doc = nlp.make_doc(act_text)

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
    print(f"\n[+] Success! Converted {len(doc_bin)} documents to: {output_file}")

    if debug_json_file:
        with open(debug_json_file, "w", encoding="utf-8") as out_json:
            json.dump(debug_output_data, out_json, ensure_ascii=False, indent=4)
        print(f"[+] Saved token verification diagnostic dump to: {debug_json_file}")


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

    print(f"Split: {len(train_data)} train / {len(dev_data)} dev "
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
    *,
    test_json: Path | None = None,
    test_text_dirs: list[Path] | None = None,
    inicjaly: bool = False,
    line_ending_mode: LineEndingMode = "lf",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    convert_exported_to_spacy(
        train_json,
        [str(d) for d in train_text_dirs],
        output_dir / "train.spacy",
        output_dir / "debug_spacy_alignment_train.json",
        inicjaly=inicjaly,
        line_ending_mode=line_ending_mode,
    )
    convert_exported_to_spacy(
        dev_json,
        [str(d) for d in dev_text_dirs],
        output_dir / "dev.spacy",
        output_dir / "debug_spacy_alignment_dev.json",
        inicjaly=inicjaly,
        line_ending_mode=line_ending_mode,
    )
    if test_json is not None:
        convert_exported_to_spacy(
            test_json,
            [str(d) for d in (test_text_dirs or dev_text_dirs)],
            output_dir / "test.spacy",
            output_dir / "debug_spacy_alignment_test.json",
            inicjaly=inicjaly,
            line_ending_mode=line_ending_mode,
        )


# --- Run ---
if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent.parent
    TRAIN_JSON = SCRIPT_DIR / "../data/exported_sentences_ls/exported_sentences_ls_merged.json"
    DEV_JSON = SCRIPT_DIR / "../data/exported_sentences_ls/wyciete_zdania_raw_2025.json"
    TEST_JSON = SCRIPT_DIR / "../data/exported_sentences_ls/wyciete_zdania_raw_2026.json"
    OUTPUT_DIR = SCRIPT_DIR / "../spacy_config"

    convert_train_dev(
        TRAIN_JSON,
        DEV_JSON,
        train_text_dirs=act_year_dirs(REPO_ROOT, ("2018", "2019", "2020", "2021")),
        dev_text_dirs=act_year_dirs(REPO_ROOT, ("2025",)),
        output_dir=OUTPUT_DIR,
        test_json=TEST_JSON,
        test_text_dirs=act_year_dirs(REPO_ROOT, ("2026",)),
    )