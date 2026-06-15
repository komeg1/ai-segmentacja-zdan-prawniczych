"""Act text normalization: CRLF -> LF (\\n) or space + offset correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

LineEndingMode = Literal["lf", "space"]

DEFAULT_EXPORTED_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "exported_sentences_ls"
)
RAW_JSON_GLOB = "wyciete_zdania_raw_*.json"


def read_raw_act_text(path: Path | str) -> str:
    """Read a _clean.txt file without line-ending translation (preserves \\r\\n)."""
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def normalize_crlf_text(raw_crlf: str, mode: LineEndingMode) -> str:
    """Replace \\r\\n with \\n or with a space."""
    if mode == "lf":
        return raw_crlf.replace("\r\n", "\n").replace("\r", "\n")
    return raw_crlf.replace("\r\n", " ")


def crlf_offset_to_normalized(offset: int, raw_crlf_text: str) -> int:
    """Shift offset from CRLF to LF or space (\\r\\n is 2 chars -> 1 char)."""
    return offset - raw_crlf_text[:offset].count("\r")


def adjust_crlf_offsets_to_normalized(
    start: int,
    end: int,
    raw_crlf_text: str,
) -> tuple[int, int]:
    return (
        crlf_offset_to_normalized(start, raw_crlf_text),
        crlf_offset_to_normalized(end, raw_crlf_text),
    )


# Backward-compatible aliases
normalize_line_endings = lambda raw: normalize_crlf_text(raw, "lf")
crlf_offset_to_lf = crlf_offset_to_normalized
adjust_crlf_offsets_to_lf = adjust_crlf_offsets_to_normalized


def prepare_act_text_from_file(
    path: Path | str,
    mode: LineEndingMode = "lf",
) -> tuple[str, str]:
    """Return (normalized_text, original_CRLF_text)."""
    raw = read_raw_act_text(path)
    return normalize_crlf_text(raw, mode), raw


def count_carriage_returns_in_export(documents: list[dict]) -> int:
    return sum(
        sentence.get("text", "").count("\r")
        for document in documents
        for sentence in document.get("sentences", [])
    )


def count_newlines_in_export(documents: list[dict]) -> int:
    return sum(
        sentence.get("text", "").count("\n")
        for document in documents
        for sentence in document.get("sentences", [])
    )


def document_uses_normalized_offsets(
    sentences: list[dict],
    normalized_text: str,
) -> bool:
    if not sentences:
        return True
    return all(
        "\r" not in sentence.get("text", "")
        and normalized_text[sentence["start_offset"] : sentence["end_offset"]]
        == sentence["text"]
        for sentence in sentences
    )


def document_uses_crlf_offsets(sentences: list[dict], raw_crlf: str) -> bool:
    if not sentences:
        return True
    return all(
        raw_crlf[sentence["start_offset"] : sentence["end_offset"]] == sentence["text"]
        for sentence in sentences
    )


def convert_document_sentences(
    sentences: list[dict],
    normalized_text: str,
    raw_crlf: str,
    mode: LineEndingMode,
) -> tuple[list[dict], bool]:
    """Recalculate offsets and text for the selected mode. Returns (sentences, changed)."""
    if document_uses_normalized_offsets(sentences, normalized_text):
        return sentences, False

    new_sentences: list[dict] = []

    if document_uses_crlf_offsets(sentences, raw_crlf):
        for sentence in sentences:
            start, end = adjust_crlf_offsets_to_normalized(
                sentence["start_offset"],
                sentence["end_offset"],
                raw_crlf,
            )
            new_sentences.append({
                "start_offset": start,
                "end_offset": end,
                "text": normalized_text[start:end],
            })
        return sorted(new_sentences, key=lambda s: s["start_offset"]), True

    # Other mode (lf <-> space) — offsets stay the same, only text changes
    other_mode: LineEndingMode = "space" if mode == "lf" else "lf"
    other_text = normalize_crlf_text(raw_crlf, other_mode)
    if document_uses_normalized_offsets(sentences, other_text):
        for sentence in sentences:
            start = sentence["start_offset"]
            end = sentence["end_offset"]
            new_sentences.append({
                "start_offset": start,
                "end_offset": end,
                "text": normalized_text[start:end],
            })
        return sorted(new_sentences, key=lambda s: s["start_offset"]), True

    # Fallback: treat stored offsets as CRLF
    for sentence in sentences:
        start, end = adjust_crlf_offsets_to_normalized(
            sentence["start_offset"],
            sentence["end_offset"],
            raw_crlf,
        )
        new_sentences.append({
            "start_offset": start,
            "end_offset": end,
            "text": normalized_text[start:end],
        })
    return sorted(new_sentences, key=lambda s: s["start_offset"]), True


# Backward-compatible alias
lf_document_sentences = lambda s, lf, raw: convert_document_sentences(s, lf, raw, "lf")


def sanitize_exported_documents(
    documents: list[dict],
    mode: LineEndingMode,
) -> tuple[list[dict], int]:
    """Align text fields with the selected mode."""
    fixed = 0
    for document in documents:
        for sentence in document.get("sentences", []):
            original = sentence.get("text", "")
            if mode == "lf":
                cleaned = original.replace("\r\n", "\n").replace("\r", "\n")
            else:
                cleaned = (
                    original.replace("\r\n", " ")
                    .replace("\n", " ")
                    .replace("\r", " ")
                )
            if cleaned != original:
                sentence["text"] = cleaned
                fixed += 1
    return documents, fixed


def validate_exported_documents(
    documents: list[dict],
    mode: LineEndingMode,
    label: str,
) -> None:
    if count_carriage_returns_in_export(documents):
        raise RuntimeError(f"{label} still contains \\r characters in text fields.")
    if mode == "space" and count_newlines_in_export(documents):
        raise RuntimeError(f"{label} still contains \\n characters (space mode).")


def iter_top_level_raw_json(exported_dir: Path) -> list[Path]:
    """Only wyciete_zdania_raw_*.json files directly in exported_sentences_ls/."""
    if not exported_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {exported_dir}")
    return sorted(p for p in exported_dir.glob(RAW_JSON_GLOB) if p.is_file())


def normalize_exported_json_file(
    json_path: Path,
    text_dirs: list[str],
    mode: LineEndingMode,
    *,
    find_clean_txt,
) -> tuple[int, int]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    updated_docs = 0
    skipped_docs = 0

    for document in data:
        act_id = document["act_id"]
        local_file_path = find_clean_txt(act_id, text_dirs)
        if not local_file_path:
            print(f"[!] {json_path.name}: missing file for {act_id}")
            continue

        normalized_text, raw_text = prepare_act_text_from_file(local_file_path, mode)
        new_sentences, changed = convert_document_sentences(
            document["sentences"],
            normalized_text,
            raw_text,
            mode,
        )
        document["sentences"] = new_sentences
        if changed:
            updated_docs += 1
        else:
            skipped_docs += 1

    data, _ = sanitize_exported_documents(data, mode)
    validate_exported_documents(data, mode, json_path.name)

    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    return updated_docs, skipped_docs


def normalize_all_top_level_raw_json(
    exported_dir: Path,
    text_dirs: list[str],
    mode: LineEndingMode,
    *,
    find_clean_txt,
) -> None:
    paths = iter_top_level_raw_json(exported_dir)
    if not paths:
        raise FileNotFoundError(
            f"No {RAW_JSON_GLOB} files found directly in {exported_dir}"
        )

    mode_label = "\\n" if mode == "lf" else "space"
    print(
        f"Replacing \\r\\n -> {mode_label} in {len(paths)} files ({exported_dir})...",
        flush=True,
    )
    for path in paths:
        updated, skipped = normalize_exported_json_file(
            path,
            text_dirs,
            mode,
            find_clean_txt=find_clean_txt,
        )
        print(f"  {path.name}: updated {updated}, already OK {skipped}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize wyciete_zdania_raw_*.json in exported_sentences_ls/ "
            "(only files in that directory, not subfolders)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("lf", "space"),
        required=True,
        help="lf: \\r\\n -> \\n; space: \\r\\n -> space",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_EXPORTED_DIR,
        help="exported_sentences_ls directory (default: spacy/data/exported_sentences_ls)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mode: LineEndingMode = args.mode
    exported_dir = args.dir.resolve()

    from convert_json_to_spacy import act_year_dirs
    from convert_ls_to_json import find_clean_txt

    repo_root = Path(__file__).resolve().parents[2]
    years = ("2016", "2018", "2019", "2020", "2021", "2025", "2026")
    text_dirs = act_year_dirs(repo_root, years)

    normalize_all_top_level_raw_json(
        exported_dir,
        text_dirs,
        mode,
        find_clean_txt=find_clean_txt,
    )


if __name__ == "__main__":
    main()
