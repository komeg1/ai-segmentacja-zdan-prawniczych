import argparse
import json
import os
import re
from pathlib import Path

from act_text_utils import (
    LineEndingMode,
    adjust_crlf_offsets_to_normalized,
    normalize_all_top_level_raw_json,
    prepare_act_text_from_file,
    sanitize_exported_documents,
    validate_exported_documents,
)


def find_clean_txt(
    act_id: str,
    text_files_directory: str | Path | list[str | Path],
) -> str | None:
    """Match file by prefix act_YYYY_N_ in one or many directories."""
    if isinstance(text_files_directory, (str, Path)):
        dirs = [str(text_files_directory)]
    else:
        dirs = [str(d) for d in text_files_directory]

    prefix = f"{act_id}_"
    for text_dir in dirs:
        if not os.path.isdir(text_dir):
            continue
        for local_file in os.listdir(text_dir):
            if local_file.startswith(prefix) and local_file.endswith("_clean.txt"):
                return os.path.join(text_dir, local_file)
    return None


def extract_raw_sentences(
    json_file_path,
    text_files_directory,
    output_json,
    *,
    mode: LineEndingMode = "lf",
):
    with open(json_file_path, "r", encoding="utf-8") as f:
        ls_data = json.load(f)

    extracted_data = []

    for task in ls_data:
        ls_text_path = task["data"]["text"]
        raw_filename = os.path.basename(ls_text_path)

        match = re.search(r"(act_\d+_\d+)", raw_filename)
        if not match:
            print(f"[!] Error: Could not find pattern in {raw_filename}.")
            continue

        act_id = match.group(1)

        local_file_path = find_clean_txt(act_id, text_files_directory)
        if not local_file_path:
            print(f"[!] Warning: Missing local text file with ID {act_id}.")
            continue

        norm_text, raw_text = prepare_act_text_from_file(local_file_path, mode)

        annotations = task["annotations"][0]["result"]

        document_result = {
            "act_id": act_id,
            "sentences": [],
        }

        for ann in annotations:
            start_crlf = ann["value"]["start"]
            end_crlf = ann["value"]["end"]
            start_char, end_char = adjust_crlf_offsets_to_normalized(
                start_crlf, end_crlf, raw_text
            )
            extracted_text = norm_text[start_char:end_char]

            document_result["sentences"].append({
                "start_offset": start_char,
                "end_offset": end_char,
                "text": extracted_text,
            })

        document_result["sentences"].sort(key=lambda s: s["start_offset"])
        mismatches = sum(
            1
            for s in document_result["sentences"]
            if norm_text[s["start_offset"] : s["end_offset"]] != s["text"]
        )
        if mismatches:
            print(f"[!] Warning: {act_id} — {mismatches} sentences do not match source file.")

        extracted_data.append(document_result)

    extracted_data, _ = sanitize_exported_documents(extracted_data, mode)
    validate_exported_documents(extracted_data, mode, "export")

    with open(output_json, "w", encoding="utf-8") as out_json:
        json.dump(extracted_data, out_json, ensure_ascii=False, indent=4)

    print(f"\n[+] Success! Saved extracted sentences to: {output_json}")
    return len(extracted_data), sum(len(d["sentences"]) for d in extracted_data)


EXPORT_YEARS = ("2016", "2018", "2019", "2020", "2021", "2025", "2026")


def export_all_years(
    *,
    data_dir: Path,
    output_dir: Path,
    repo_root: Path,
    mode: LineEndingMode,
) -> None:
    from convert_json_to_spacy import act_year_dirs

    output_dir.mkdir(parents=True, exist_ok=True)
    total_docs = 0
    total_sents = 0

    for year in EXPORT_YEARS:
        ls_json = data_dir / f"gold_{year}.json"
        output_json = output_dir / f"wyciete_zdania_raw_{year}.json"
        if not ls_json.is_file():
            print(f"[!] Skipping {year}: missing {ls_json.name}")
            continue

        text_dirs = act_year_dirs(repo_root, (year,))
        if not text_dirs:
            print(f"[!] Skipping {year}: missing directory with _clean.txt")
            continue

        print(f"\n=== {year} (mode={mode}) ===", flush=True)
        print(f"  gold:   {ls_json}", flush=True)
        print(f"  texts: {text_dirs}", flush=True)
        print(f"  out:    {output_json}", flush=True)

        docs, sents = extract_raw_sentences(
            ls_json,
            text_dirs,
            output_json,
            mode=mode,
        )
        total_docs += docs
        total_sents += sents
        print(f"  -> {docs} acts, {sents} sentences", flush=True)

    print(
        f"\n[+] Export finished: {total_docs} acts, {total_sents} sentences "
        f"(mode={mode}) -> {output_dir}",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export from Label Studio -> wyciete_zdania_raw JSON.")
    parser.add_argument(
        "--export-all-years",
        action="store_true",
        help="Export gold_YYYY.json -> wyciete_zdania_raw_YYYY.json for all years",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (--export-all-years); default exported_sentences_ls/",
    )
    parser.add_argument(
        "--normalize-existing",
        action="store_true",
        help="Normalize wyciete_zdania_raw_*.json (top-level exported_sentences_ls/ only)",
    )
    parser.add_argument(
        "--mode",
        choices=("lf", "space"),
        default="lf",
        help="lf: \\r\\n -> \\n; space: \\r\\n -> space (default lf)",
    )
    parser.add_argument("--ls-json", default="../data/gold_2016.json")
    parser.add_argument("--txt-dir", default="../../legal-text-downloader/data/acts/2016/")
    parser.add_argument(
        "--output",
        default="../data/exported_sentences_ls/wyciete_zdania_raw_2016.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    mode: LineEndingMode = args.mode

    data_dir = script_dir.parent / "data"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else script_dir.parent / "data" / "exported_sentences_ls"
    )

    if args.export_all_years:
        export_all_years(
            data_dir=data_dir,
            output_dir=output_dir,
            repo_root=repo_root,
            mode=mode,
        )
    elif args.normalize_existing:
        from convert_json_to_spacy import act_year_dirs

        years = ("2016", "2018", "2019", "2020", "2021", "2025", "2026")
        text_dirs = act_year_dirs(repo_root, years)
        normalize_all_top_level_raw_json(
            script_dir.parent / "data" / "exported_sentences_ls",
            text_dirs,
            mode,
            find_clean_txt=find_clean_txt,
        )
    else:
        extract_raw_sentences(
            args.ls_json,
            args.txt_dir,
            args.output,
            mode=mode,
        )
