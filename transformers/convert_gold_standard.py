"""
Converts Label Studio export (gold standard) to dataset format
compatible with prepare_dataset.py:
    {"tokens": [...], "sbd_labels": [...]}

Usage:
    python convert_gold_standard.py \
        --ls_json    data/gold_standard.json \
        --txt_dir    data/acts/2026 \
        --output     data/gold_dataset
"""

import argparse
import json
import os
import re

from datasets import Dataset
import regex


def tokenize(text: str) -> list:
    return regex.findall(r"\p{L}+|\d+|[^\s\p{L}\d]", text)


def find_local_file(file_upload: str, txt_dir: str) -> str | None:
    raw_filename = os.path.basename(file_upload)
    match = re.search(r"(act_\d+_\d+)", raw_filename)
    if not match:
        return None
    act_id = match.group(1)
    for fname in os.listdir(txt_dir):
        if act_id in fname and fname.endswith("_clean.txt"):
            return os.path.join(txt_dir, fname)
    return None


def task_to_samples(task: dict, txt_dir: str) -> list:
    """Converts a single Label Studio task into a list of training windows.

    Each annotated sentence (segment) is a token sequence ending with label 1.
    Tokens between sentences (if any) receive label 0.
    """
    file_upload = task.get("file_upload", "")
    local_path = find_local_file(file_upload, txt_dir)

    if not local_path:
        print(f"[!] Missing file for: {file_upload}")
        return []

    with open(local_path, "r", encoding="utf-8", newline="") as f:
        raw_text = f.read()

    annotations = task.get("annotations", [])
    if not annotations:
        return []

    results = annotations[0].get("result", [])
    if not results:
        return []

    segments = sorted(results, key=lambda r: r["value"]["start"])

    samples = []
    all_tokens = []
    all_labels = []

    for seg in segments:
        start = seg["value"]["start"]
        end = seg["value"]["end"]

        seg_text = raw_text[start:end].strip()
        if not seg_text:
            continue

        tokens = tokenize(seg_text)
        if not tokens:
            continue

        labels = [0] * len(tokens)
        labels[-1] = 1

        all_tokens.extend(tokens)
        all_labels.extend(labels)

    if not all_tokens:
        return []

    boundary_positions = [i for i, l in enumerate(all_labels) if l == 1]

    if not boundary_positions:
        return []

    window_tokens = []
    window_labels = []
    prev_boundary = 0

    for bp in boundary_positions:
        chunk_tokens = all_tokens[prev_boundary : bp + 1]
        chunk_labels = all_labels[prev_boundary : bp + 1]

        window_tokens.extend(chunk_tokens)
        window_labels.extend(chunk_labels)

        if len(window_tokens) > 450:
            if len(window_tokens) >= 10:
                samples.append(
                    {
                        "tokens": window_tokens,
                        "sbd_labels": window_labels,
                    }
                )
            window_tokens = []
            window_labels = []

        prev_boundary = bp + 1

    if 10 <= len(window_tokens) <= 450:
        samples.append({"tokens": window_tokens, "sbd_labels": window_labels})
    elif len(window_tokens) > 0 and samples:
        if len(samples[-1]["tokens"]) + len(window_tokens) <= 500:
            samples[-1]["tokens"].extend(window_tokens)
            samples[-1]["sbd_labels"].extend(window_labels)

    return samples


def print_stats(data: list) -> None:
    total_tokens = sum(len(s["tokens"]) for s in data)
    total_ends = sum(sum(s["sbd_labels"]) for s in data)
    ratio = total_ends / total_tokens * 100 if total_tokens else 0
    print(f"  Windows:             {len(data):,}")
    print(f"  Tokens:              {total_tokens:,}")
    print(f"  Sentence boundaries: {total_ends:,} ({ratio:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Converts Label Studio export into HuggingFace dataset format."
    )
    parser.add_argument(
        "--ls_json", required=True, help="Path to Label Studio JSON export"
    )
    parser.add_argument(
        "--txt_dir", required=True, help="Directory containing _clean.txt files"
    )
    parser.add_argument("--output", required=True, help="Output dataset directory path")
    args = parser.parse_args()

    with open(args.ls_json, "r", encoding="utf-8") as f:
        ls_data = json.load(f)

    print(f"Loaded {len(ls_data)} tasks from Label Studio.")

    all_samples = []
    for task in ls_data:
        samples = task_to_samples(task, args.txt_dir)
        all_samples.extend(samples)
        if samples:
            file_name = os.path.basename(task.get("file_upload", ""))
            print(f"  {file_name[:60]} -> {len(samples)} windows")

    print("\nGold standard statistics:")
    print_stats(all_samples)

    dataset = Dataset.from_list(all_samples)
    os.makedirs(args.output, exist_ok=True)
    dataset.save_to_disk(args.output)
    print(f"\nDataset successfully saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
