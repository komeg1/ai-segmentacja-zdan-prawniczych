import json
import os
from pathlib import Path

from act_text_utils import count_carriage_returns_in_export, sanitize_exported_documents

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "../data/exported_sentences_ls"
OUTPUT_FILE = INPUT_DIR / "exported_sentences_ls_merged.json"


def merge_exported_sentences(input_dir: Path, output_file: Path) -> dict:
    merged = []
    source_files = sorted(
        p for p in input_dir.glob("wyciete_zdania_raw_*.json")
        if not p.name.endswith(("_2025.json", "_2026.json"))  # dev/test — not for train
    )

    if not source_files:
        raise FileNotFoundError(
            f"No wyciete_zdania_raw_*.json files in directory: {input_dir}"
        )

    for source_file in source_files:
        with open(source_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected a list of documents in file: {source_file.name}")

        for doc in data:
            if "sentences" not in doc:
                raise ValueError(
                    f"Missing 'sentences' key in {source_file.name} "
                    f"(act_id={doc.get('act_id', '?')})"
                )

        merged.extend(data)
        print(f"  + {source_file.name}: {len(data)} documents, "
              f"{sum(len(d['sentences']) for d in data)} sentences")

    merged, fixed = sanitize_exported_documents(merged, "lf")
    if fixed:
        print(f"  [!] Removed \\r from {fixed} text fields during merge.", flush=True)

    remaining_cr = count_carriage_returns_in_export(merged)
    if remaining_cr:
        raise RuntimeError(
            f"merged JSON contains \\r={remaining_cr} in text fields — "
            "run convert_ls_to_json.py --normalize-existing"
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)

    stats = {
        "source_files": len(source_files),
        "documents": len(merged),
        "sentences": sum(len(d["sentences"]) for d in merged),
    }
    return stats


if __name__ == "__main__":
    input_dir = INPUT_DIR.resolve()
    output_file = OUTPUT_FILE.resolve()

    print(f"Merging files from: {input_dir}")
    stats = merge_exported_sentences(input_dir, output_file)

    print(f"\n[+] Saved merged file: {output_file}")
    print(f"    Source files: {stats['source_files']}")
    print(f"    Documents:    {stats['documents']}")
    print(f"    Sentences:    {stats['sentences']}")
