import glob
import os
import nltk
from datasets import Dataset, DatasetDict
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer
import regex as re

nltk.download("punkt", quiet=True)


# ── Stałe ─────────────────────────────────────────────────────────────────────

PROTECTED_TERMS = {
    "art",
    "ust",
    "pkt",
    "poz",
    "nr",
    "dz",
    "u",
    "r",
    "tj",
    "urz",
    "str",
    "lp",
    "par",
    "al",
    "lit",
    "m",
    "zd",
    "rozdz",
    "proc",
    "tys",
    "mln",
    "mld",
    "godz",
    "min",
    "sek",
    "ul",
    "pl",
    "późn",
    "zm",
    "sygn",
    "kt",
    "ww",
}

ARTICLE_HEADER_RE = re.compile(
    r"""
    ^
    \s* (?:„|")? \s*
    (
        (?:art | \u00a7)
        \s* \.? \s* \d+\s*[a-z]? \s* \.
        (?: \s* \d+\s*[a-z]? \s* \. )*
    |
        \d+\s*[a-z]? \s+ \w+ \s* \.
    |
        \d+\s*[a-z]? \s* \. \s* \d+\s*[a-z]? \s* \.
    )
    \s* (?:"|")? \s* $
    """,
    re.IGNORECASE | re.VERBOSE,
)

ENDS_WITH_HEADER_RE = re.compile(
    r"""
    (?:art | \u00a7) \s* \.? \s* \d+\s*[a-z]? \s* \.
    (?: \s* \d+\s*[a-z]? \s* \. )*
    \s* $
    """,
    re.IGNORECASE | re.VERBOSE,
)

CHAPTER_HEADER_RE = re.compile(
    r"^\s*(rozdzia[łl]\s+\d+|dzia[łl]\s+[IVXLCDM]+)(\s+.*)?$",
    re.IGNORECASE,
)

LIST_ITEM_RE = re.compile(
    r"""
    ^
    \s* (?:„|")? \s*
    (?:
        \d+\s*\w*\s*\)
        |
        [a-z]{1,2}\s*\)
    )
    \s*
    """,
    re.IGNORECASE | re.VERBOSE,
)

PREV_ENDS_WITH_SENTENCE_RE = re.compile(
    r"\p{L}\s*\.\s*$",
    re.UNICODE,
)

ARTIFACT_RE = re.compile(
    r"^(zm\.?\)?|z\s+po\u017an\.?|\d+,\s*str\..{0,30}|\(uchylony\)\.?)\s*$",
    re.IGNORECASE,
)

DZIENNIK_RE = re.compile(r"^DZIENNIK\s+USTAW", re.IGNORECASE)

ACT_TYPE_RE = re.compile(
    r"^\s*(ROZPORZĄDZENIE|OBWIESZCZENIE|USTAWA|UCHWAŁA|DEKRET|OŚWIADCZENIE|WYROK|POSTANOWIENIE|ZARZĄDZENIE)(\s+.*)?$",
    re.IGNORECASE,
)

EARLY_BODY_START_RE = re.compile(
    r"(?:(?<=[.;])\s*|\A\s*)(na\s+podstawie|zarządza\s+się|uchwala\s+się|ogłasza\s+się)\b",
    re.IGNORECASE,
)

BODY_START_RE = re.compile(
    r"^\s*(na\s+podstawie|zarządza\s+się|uchwala\s+się|ogłasza\s+się)\b",
    re.IGNORECASE,
)

ATTACHMENT_RE = re.compile(r"^\s*Załącznik\s+do\s+", re.IGNORECASE)

ONLY_PUNCT_OR_FORM_RE = re.compile(r'^[…\s\.\,\-\_\/\(\)\§"„"]+$')

STRIP_LEAD_NEWLINE_RE = re.compile(r"^\n\s*")

STANDALONE_NUM_RE = re.compile(r"^\s*\(?\s*\d+\s*[a-z]?\s*\.\s*\)?\s*$", re.IGNORECASE)

ENDS_WITH_R_RE = re.compile(r"\br\s*\.\s*$")

SENT_SEP = " [SPLIT_MARKER] "

TOKENIZE_RE = re.compile(r"\p{L}+|\d+|[^\s\p{L}\d]")


def preprocess_text(text: str) -> str:
    text = re.sub(r"(\.{3,}|…+)", r" \1 ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    lines = text.splitlines()
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    return "\n".join(lines)


def tokenize(text: str) -> list:
    return TOKENIZE_RE.findall(text)


def build_punkt_tokenizer() -> PunktSentenceTokenizer:
    params = PunktParameters()
    for abbrev in PROTECTED_TERMS:
        params.abbrev_types.add(abbrev)
    return PunktSentenceTokenizer(params)


_PUNKT = build_punkt_tokenizer()


def proactive_split(text: str) -> list:
    lines = text.splitlines()
    processed_lines = []

    for line in lines:
        matches = list(EARLY_BODY_START_RE.finditer(line))
        if matches:
            last_match = matches[-1]
            cutoff = last_match.start()
            if cutoff > 0:
                part1 = line[:cutoff].strip()
                part2 = line[cutoff:].strip()
                if part1:
                    processed_lines.append(part1)
                if part2:
                    processed_lines.append(part2)
            else:
                processed_lines.append(line)
        else:
            processed_lines.append(line)

    text_preprocessed = "\n".join(processed_lines)
    text_for_punkt = text_preprocessed.replace("\n", SENT_SEP)
    segments = _PUNKT.tokenize(text_for_punkt)

    result = []
    for seg in segments:
        parts = [p.strip() for p in seg.split("[SPLIT_MARKER]") if p.strip()]
        result.extend(parts)
    return result


def clean_sentences(sentences: list) -> list:
    cleaned = []
    for s in sentences:
        s_stripped = s.strip()

        if cleaned and re.match(r'^[”"’\'\s]*[\;\.\,]\s*$', s_stripped):
            cleaned[-1] = cleaned[-1] + " " + s_stripped
            continue

        if (
            len(s_stripped) >= 2
            and not ARTIFACT_RE.match(s_stripped)
            and not ONLY_PUNCT_OR_FORM_RE.match(s_stripped)
        ):
            cleaned.append(s_stripped)

    return cleaned


def merge_article_headers(sentences: list) -> list:
    if not sentences:
        return []

    result = []
    n = len(sentences)
    i = 0

    while i < n:
        sent = sentences[i]
        sent_stripped = STRIP_LEAD_NEWLINE_RE.sub("", sent).strip()

        if DZIENNIK_RE.match(sent_stripped):
            merged_sent = sent
            j = i + 1
            while j < n:
                if sentences[j].strip():
                    next_seg = sentences[j]
                    merged_sent += " " + next_seg
                    if ENDS_WITH_R_RE.search(next_seg):
                        i = j
                        break
                j += 1
            result.append(merged_sent)
            i += 1
            continue

        is_act_start = bool(ACT_TYPE_RE.match(sent_stripped))
        is_attachment_start = (
            ATTACHMENT_RE.match(sent_stripped)
            and i + 1 < n
            and ACT_TYPE_RE.match(sentences[i + 1].strip())
        )

        if is_act_start or is_attachment_start:
            merged_act = sent_stripped
            j = i + 1

            while j < n:
                next_seg = sentences[j].strip()
                if not next_seg:
                    j += 1
                    continue

                if (
                    ARTICLE_HEADER_RE.match(next_seg)
                    or CHAPTER_HEADER_RE.match(next_seg)
                    or BODY_START_RE.match(next_seg)
                    or LIST_ITEM_RE.match(next_seg)
                ):
                    break

                if ACT_TYPE_RE.match(next_seg) and not BODY_START_RE.search(next_seg):
                    break

                if BODY_START_RE.search(next_seg):
                    match = BODY_START_RE.search(next_seg)
                    cutoff_idx = match.start()
                    part_before = next_seg[:cutoff_idx].strip()
                    if part_before:
                        merged_act += " " + part_before
                    break

                merged_act += " " + next_seg
                i = j
                j += 1

            result.append(merged_act)
            i += 1
            continue

        prev = result[-1].strip() if result else ""

        is_article_header = bool(
            ARTICLE_HEADER_RE.match(sent_stripped)
            and not CHAPTER_HEADER_RE.match(sent_stripped)
        )
        ends_with_header = bool(ENDS_WITH_HEADER_RE.search(sent_stripped))
        is_list_item = bool(LIST_ITEM_RE.match(sent_stripped))
        is_standalone_number = bool(
            STANDALONE_NUM_RE.match(sent_stripped)
            or re.match(r"^\s*§\s*\d+\s*[a-z]?\s*\.\s*$", sent_stripped, re.IGNORECASE)
        )

        if (
            is_article_header
            or ends_with_header
            or is_list_item
            or is_standalone_number
        ):
            next_idx = i + 1
            while next_idx < n and not sentences[next_idx].strip():
                next_idx += 1

            if next_idx < n:
                next_content = sentences[next_idx].strip()

                if is_standalone_number:
                    if not (
                        ARTICLE_HEADER_RE.match(next_content)
                        or LIST_ITEM_RE.match(next_content)
                    ):
                        result.append(sent_stripped + " " + next_content)
                        i = next_idx + 1
                        continue

                if is_article_header and STANDALONE_NUM_RE.match(next_content):
                    result.append(sent)
                    i += 1
                    continue

                if is_list_item and (
                    ARTICLE_HEADER_RE.match(next_content)
                    or STANDALONE_NUM_RE.match(next_content)
                ):
                    result.append(sent)
                    i += 1
                    continue

                if next_content.startswith("„") and len(sent_stripped) <= 15:
                    result.append(sent)
                    i += 1
                    continue

                result.append(sent + " " + next_content)
                i = next_idx + 1
                continue

        result.append(sent)
        i += 1

    return result


def find_sentence_ends(tokens: list, protected: set) -> list:
    ends = []
    n = len(tokens)

    letters_count = sum(1 for t in tokens if len(t) == 1 and t.isalpha())
    non_empty = [t for t in tokens if t.strip()]
    is_spaced_text = n > 5 and non_empty and (letters_count / len(non_empty)) > 0.7

    for idx in range(n):
        token = tokens[idx]

        if token == ":":
            if idx + 1 < n:
                nxt = tokens[idx + 1]

                is_next_list_item = (
                    idx + 2 < n
                    and (nxt.isdigit() or (len(nxt) == 1 and nxt.isalpha()))
                    and tokens[idx + 2] == ")"
                )

                if (
                    nxt == "\u00a7"
                    or nxt.lower() == "art"
                    or nxt in {"\u201e", '"', "\u2019", "'", "„", "\u201d"}
                    or nxt in {"-", "\u2013", "\u2014"}
                    or (nxt[0].isupper() and nxt.lower() not in protected)
                    or (nxt.isdigit() and idx + 2 < n and tokens[idx + 2] == ")")
                ):
                    ends.append(idx)
            continue

        if token in {",", ";"}:
            if (
                idx + 2 < n
                and (tokens[idx + 1].isdigit() or len(tokens[idx + 1]) == 1)
                and tokens[idx + 2] == ")"
            ):
                ends.append(idx)
                continue

            if idx + 1 < n and tokens[idx + 1] == "§":
                continue
            is_end_of_stream = all(not t.strip() for t in tokens[idx + 1 :])
            if is_end_of_stream:
                ends.append(idx)
            continue

        if token in {".", "?", "!"}:
            if idx > 0:
                prev = tokens[idx - 1].lower()
                if len(prev) == 1 and prev.isalpha():
                    continue
                if prev in protected:
                    continue
                if idx + 1 < n and (
                    tokens[idx + 1].isdigit() or tokens[idx + 1].lower() in protected
                ):
                    continue
                if tokens[idx - 1].isdigit() and idx < n - 1:
                    continue
            ends.append(idx)
            continue

        if token == ")":
            continue

        if token == "[" and idx + 1 < n and tokens[idx + 1].upper() == "TABELA":
            if idx > 0:
                ends.append(idx - 1)
            continue

        if token == "]" and idx > 0 and tokens[idx - 1].upper() == "TABELA":
            ends.append(idx)
            continue

    if is_spaced_text and n - 1 not in ends:
        ends.append(n - 1)

    return ends


def process_files(files: list, window_size: int = 10) -> list:
    all_samples = []

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = preprocess_text(f.read())

            segments = proactive_split(raw)
            sentences = clean_sentences(segments)
            sentences = merge_article_headers(sentences)

            for i in range(0, len(sentences), window_size):
                window = sentences[i : i + window_size]
                combined_tokens = []
                combined_labels = []

                for sent in window:
                    tokens = tokenize(sent)
                    if not tokens:
                        continue

                    labels = [0] * len(tokens)
                    end_indices = find_sentence_ends(tokens, PROTECTED_TERMS)

                    if end_indices:
                        for ei in end_indices:
                            labels[ei] = 1
                    else:
                        labels[-1] = 1

                    combined_tokens.extend(tokens)
                    combined_labels.extend(labels)

                n = len(combined_tokens)
                if 10 < n < 500:
                    all_samples.append(
                        {
                            "tokens": combined_tokens,
                            "sbd_labels": combined_labels,
                        }
                    )

        except Exception as e:
            print(f"Błąd w pliku {file_path}: {e}")

    return all_samples


def print_stats(name: str, data: list) -> None:
    total_tokens = sum(len(s["tokens"]) for s in data)
    total_ends = sum(sum(s["sbd_labels"]) for s in data)
    ratio = total_ends / total_tokens * 100 if total_tokens else 0
    print(
        f"  {name}: {len(data):,} okien | {total_tokens:,} tokenów | "
        f"{total_ends:,} granic zdań ({ratio:.1f}%)"
    )


def main():
    BASE_DIR = "."
    SAVE_PATH = "data/final_dataset_v17"

    train_paths = []
    for year in range(2015, 2025):
        train_paths += glob.glob(
            os.path.join(BASE_DIR, f"data/acts/{year}/*_clean.txt")
        )

    val_paths = glob.glob(os.path.join(BASE_DIR, "data/acts/2025/*_clean.txt"))

    print(f"Znaleziono plików: train={len(train_paths)}, val={len(val_paths)}")
    print("Generowanie Datasetu V17..")

    train_data = process_files(train_paths, window_size=10)
    val_data = process_files(val_paths, window_size=10)

    print("\nStatystyki:")
    print_stats("Train", train_data)
    print_stats("Val", val_data)

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_data),
            "test": Dataset.from_list(val_data),
        }
    )

    os.makedirs(SAVE_PATH, exist_ok=True)
    dataset_dict.save_to_disk(SAVE_PATH)

    print(f"\nDataset V17 zapisany w: {os.path.abspath(SAVE_PATH)}")


if __name__ == "__main__":
    main()
