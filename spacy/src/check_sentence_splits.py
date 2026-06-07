import json
import os
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent

VALID_LOWERCASE_START = re.compile(
    r"^([a-ząćęłńóśźż]\)|\d+[\).]|[a-ząćęłńóśźż],|w zw\.|tj\.|np\.|itd\.|itp\.|tzw\.|m\. in\.|art\.|ust\.|pkt\.|lit\.|§\s*\d)"
)


def _acts_root() -> Path:
    gold = BASE / "legal-text-downloader/data/acts_for_gold"
    full = BASE / "legal-text-downloader/data/acts"
    if gold.is_dir() and any(gold.iterdir()):
        return gold
    return full


def find_source(act_id: str) -> Path | None:
    year = act_id.split("_")[1]
    acts_dir = _acts_root() / year
    if not acts_dir.exists():
        return None
    prefix = f"{act_id}_"
    for filename in os.listdir(acts_dir):
        if filename.startswith(prefix) and filename.endswith("_clean.txt"):
            return acts_dir / filename
    return None


def is_legit_lowercase_start(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped or stripped[0].isupper() or stripped[0].isdigit():
        return True
    return bool(VALID_LOWERCASE_START.match(stripped))


def ends_mid_word(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped[-1] in '.:;!?»"”\')':
        return False
    if re.search(r"[a-ząćęłńóśźż]$", stripped):
        return True
    return False


def analyze_file(path: Path, label: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    issues = {
        "offset_mismatch": [],
        "missing_source": [],
        "lowercase_start": [],
        "ends_mid_word": [],
        "gap_text_mismatch": [],
        "overlap": [],
    }

    total_sents = 0

    for doc in data:
        act_id = doc["act_id"]
        sentences = doc["sentences"]
        total_sents += len(sentences)
        source_path = find_source(act_id)

        raw_text = None
        if source_path:
            with open(source_path, encoding="utf-8", newline="") as f:
                raw_text = f.read()
        else:
            issues["missing_source"].append(act_id)

        for idx, sentence in enumerate(sentences):
            text = sentence["text"]
            start = sentence["start_offset"]
            end = sentence["end_offset"]

            if raw_text is not None:
                extracted = raw_text[start:end]
                if extracted != text:
                    issues["offset_mismatch"].append({
                        "act_id": act_id,
                        "idx": idx + 1,
                        "start": start,
                        "end": end,
                        "stored": text[:80],
                        "from_file": extracted[:80],
                    })

            if idx > 0 and not is_legit_lowercase_start(text):
                issues["lowercase_start"].append({
                    "act_id": act_id,
                    "idx": idx + 1,
                    "text": text[:100],
                })

            if ends_mid_word(text):
                issues["ends_mid_word"].append({
                    "act_id": act_id,
                    "idx": idx + 1,
                    "text": text[-60:],
                })

            if raw_text is not None and idx < len(sentences) - 1:
                next_sentence = sentences[idx + 1]
                gap = raw_text[end:next_sentence["start_offset"]]
                if gap and not gap.isspace():
                    issues["gap_text_mismatch"].append({
                        "act_id": act_id,
                        "idx": idx + 1,
                        "gap": repr(gap[:40]),
                        "end_text": text[-30:],
                        "next_start": next_sentence["text"][:30],
                    })
                if next_sentence["start_offset"] < end:
                    issues["overlap"].append({
                        "act_id": act_id,
                        "idx": idx + 1,
                        "end": end,
                        "next_start": next_sentence["start_offset"],
                    })

    print(f"\n=== {label} ===")
    print(f"Dokumenty: {len(data)}, Zdania: {total_sents}")
    print(f"Brak pliku zrodlowego: {len(issues['missing_source'])}")
    print(f"Niezgodnosc offsetow ze zrodlem: {len(issues['offset_mismatch'])}")
    print(f"Podejrzany start (mala litera): {len(issues['lowercase_start'])}")
    print(f"Podejrzane zakonczenie (w polowie slowa): {len(issues['ends_mid_word'])}")
    print(f"Nie-whitespace w luce miedzy zdaniami: {len(issues['gap_text_mismatch'])}")
    print(f"Nakladanie sie offsetow: {len(issues['overlap'])}")

    for key in ("offset_mismatch", "lowercase_start", "ends_mid_word", "gap_text_mismatch"):
        examples = issues[key]
        if examples:
            print(f"\n  Przyklady [{key}] (max 5):")
            for example in examples[:5]:
                print(f"    {example}")

    return issues


def has_word_chars(text: str) -> bool:
    return bool(re.search(r"[A-Za-z\u0100-\u024f]", text))


def analyze_real_errors(path: Path, label: str) -> None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    missed_text = []
    overlaps = []
    split_mid_word = []

    for doc in data:
        act_id = doc["act_id"]
        source_path = find_source(act_id)
        if not source_path:
            continue
        with open(source_path, encoding="utf-8", newline="") as f:
            raw_text = f.read()

        sentences = doc["sentences"]
        for idx in range(len(sentences) - 1):
            current = sentences[idx]
            nxt = sentences[idx + 1]
            gap = raw_text[current["end_offset"]:nxt["start_offset"]]
            gap_stripped = gap.strip()

            if nxt["start_offset"] < current["end_offset"]:
                overlaps.append({
                    "act_id": act_id,
                    "idx": idx + 1,
                    "end": current["end_offset"],
                    "next_start": nxt["start_offset"],
                })
            elif gap_stripped and has_word_chars(gap_stripped):
                missed_text.append({
                    "act_id": act_id,
                    "idx": idx + 1,
                    "gap": gap_stripped[:100],
                    "prev_end": current["text"][-50:],
                    "next_start": nxt["text"][:50],
                })

            prev_end = current["text"].rstrip()
            next_start = nxt["text"].lstrip()
            if (
                prev_end
                and next_start
                and prev_end[-1].isalpha()
                and next_start[0].islower()
                and not gap_stripped
                and not VALID_LOWERCASE_START.match(next_start)
            ):
                split_mid_word.append({
                    "act_id": act_id,
                    "idx": idx + 1,
                    "prev_end": prev_end[-60:],
                    "next_start": next_start[:60],
                })

    print(f"\n--- Rzeczywiste bledy: {label} ---")
    print(f"Pominiete fragmenty (luka ze slowami): {len(missed_text)}")
    print(f"Nakladanie offsetow: {len(overlaps)}")
    print(f"Podzial w srodku slowa/zdania: {len(split_mid_word)}")

    if missed_text:
        print("\n  Pominiete fragmenty (max 10):")
        for example in missed_text[:10]:
            print(f"    {example['act_id']} zd.{example['idx']}")
            print(f"      GAP: {example['gap']!r}")
            print(f"      poprz: ...{example['prev_end']}")
            print(f"      nast:  {example['next_start']}...")

    if split_mid_word:
        print("\n  Podzial w srodku slowa (max 10):")
        for example in split_mid_word[:10]:
            print(f"    {example['act_id']} zd.{example['idx']}")
            print(f"      ...{example['prev_end']} | {example['next_start']}...")


if __name__ == "__main__":
    files = sys.argv[1:] or [
        BASE / "spacy/data/exported_sentences_ls/exported_sentences_ls_merged.json",
        BASE / "spacy/data/exported_sentences_ls/wyciete_zdania_raw_2026.json",
    ]
    labels = ["TRAIN (merged)", "DEV (2026)"]
    for path, label in zip(files, labels):
        analyze_file(Path(path), label)
        analyze_real_errors(Path(path), label)
