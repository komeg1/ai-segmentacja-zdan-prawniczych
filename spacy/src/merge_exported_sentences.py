import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "../data/exported_sentences_ls"
OUTPUT_FILE = INPUT_DIR / "exported_sentences_ls_merged.json"


def merge_exported_sentences(input_dir: Path, output_file: Path) -> dict:
    merged = []
    source_files = sorted(
        p for p in input_dir.glob("wyciete_zdania_raw_*.json")
        if not p.name.endswith("_2026.json")  # dev out-of-time — nie do train
    )

    if not source_files:
        raise FileNotFoundError(
            f"Brak plików wyciete_zdania_raw_*.json w katalogu: {input_dir}"
        )

    for source_file in source_files:
        with open(source_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Oczekiwano listy dokumentów w pliku: {source_file.name}")

        for doc in data:
            if "sentences" not in doc:
                raise ValueError(
                    f"Brak klucza 'sentences' w {source_file.name} "
                    f"(act_id={doc.get('act_id', '?')})"
                )

        merged.extend(data)
        print(f"  + {source_file.name}: {len(data)} dokumentów, "
              f"{sum(len(d['sentences']) for d in data)} zdań")

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

    print(f"Łączenie plików z: {input_dir}")
    stats = merge_exported_sentences(input_dir, output_file)

    print(f"\n[+] Zapisano połączony plik: {output_file}")
    print(f"    Pliki źródłowe: {stats['source_files']}")
    print(f"    Dokumenty:      {stats['documents']}")
    print(f"    Zdania:         {stats['sentences']}")
