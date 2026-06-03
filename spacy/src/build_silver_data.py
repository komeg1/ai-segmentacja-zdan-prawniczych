import os
import glob
import nltk
import regex as re
from datasets import Dataset, DatasetDict
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

nltk.download("punkt", quiet=True)


# ── Stałe ─────────────────────────────────────────────────────────────────────

PROTECTED_TERMS = {
    "art", "ust", "pkt", "poz", "nr", "dz", "u", "r", "tj", "urz",
    "str", "lp", "par", "al", "lit", "m", "zd", "rozdz", "proc",
    "tys", "mln", "mld", "godz", "min", "sek", "ul", "pl",
}

# Nagłówek artykułu/ustępu/punktu wyliczenia — nie jest samodzielnym zdaniem.
# Obsługuje spacje między tokenami (efekt tokenizacji regex):
#   "Art. 3."  "Art. 3. 1."  "§ 5. 1."  "§ 4 a ."  "Art . 354 a ."
#   "8. 1."    "8b. 1."      "1."        "2a."
#   "1 )"      "2 )"         "a )"       "b )"
HEADER_RE = re.compile(
    r"""
    ^
    (
        # Art. X.  /  Art. X. Y.  /  § X. Y.  /  § 4 a .  /  Art . 354 a .
        (?:art | \u00a7 | rozdzia\u0142 | rozdz)
        \s* \.? \s* \d+ \s* \w* \s* \.
        (?: \s* \d+ \s* \w* \s* \. )*
    |
        # "14a. 1."  "96b. 1."  "8. 1." — dwa człony bez słowa Art.
        \d+ \s* \w* \s* \. \s* \d+ \s* \w* \s* \.
    |
        # Pojedyncza numeracja: "1."  "2a."  "10."
        \d+ \s* \w* \s* \.
    |
        # Numer punktu wyliczenia: "1 )"  "2 )"  "10 )"  "a )"  "b )"
        [\da-z] \w* \s* \)
    )
    \s* $
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Artefakty PDF — krótkie śmieci do usunięcia
ARTIFACT_RE = re.compile(
    r"^(zm\.?\)?|z\s+po\u017an\.?|\d+,\s*str\..{0,30}|\(uchylony\)\.?)\s*$",
    re.IGNORECASE,
)


# ── Pomocnicze ────────────────────────────────────────────────────────────────

def remove_headers(text: str) -> str:
    """Usuwa nagłówki i metadane dzienników ustaw."""
    text = re.sub(r"DZIENNIK USTAW.*?\d{4}\s*r\.", "", text)
    text = re.sub(r"USTAWA\s+z\s+dnia\s+\d+.*?\d{4}\s*r\.", "", text)
    text = re.sub(r"ROZPORZ\u0104DZENIE\s+.*?\d{4}\s*r\.", "", text)
    return text


def tokenize(text: str) -> list:
    """Tokenizacja wspierająca polskie znaki."""
    return re.findall(r"\p{L}+|\d+|[^\s\p{L}\d]", text)


def build_punkt_tokenizer() -> PunktSentenceTokenizer:
    """Buduje tokenizer Punkt z listą skrótów prawniczych."""
    params = PunktParameters()
    for abbrev in PROTECTED_TERMS:
        params.abbrev_types.add(abbrev)
    return PunktSentenceTokenizer(params)


_PUNKT = build_punkt_tokenizer()


def proactive_split(text: str) -> list:
    return _PUNKT.tokenize(text)


def clean_sentences(sentences: list) -> list:
    """Usuwa puste i zbyt krótkie segmenty."""
    return [s.strip() for s in sentences if len(s.strip()) >= 3]


def merge_article_headers(sentences: list) -> list:
    """
    1. Usuwa artefakty PDF.
    2. Scala nagłówki (Art., parz., numery punktow) z nastepujacym zdaniem.
       Petla max 5 przejsc obsluguje wieloczlonowe naglowki:
         ["Art. 64.", "1.", "W przypadku..."] -> ["Art. 64. 1. W przypadku..."]
         ["1 )", "grozbie uzycia..."]         -> ["1 ) grozbie uzycia..."]
    """
    cleaned = [s for s in sentences if not ARTIFACT_RE.match(s.strip())]

    def one_pass(sents):
        result = []
        i = 0
        while i < len(sents):
            sent = sents[i].strip()
            next_nonempty = None
            j = i + 1
            while j < len(sents) and not sents[j].strip():
                j += 1
            if j < len(sents):
                next_nonempty = j
            if HEADER_RE.match(sent) and next_nonempty is not None:
                result.append(sent + " " + sents[next_nonempty].strip())
                i = next_nonempty + 1
            else:
                result.append(sent)
                i += 1
        return result

    prev = cleaned
    for _ in range(5):
        curr = one_pass(prev)
        if curr == prev:
            break
        prev = curr

    return prev


def find_sentence_ends(tokens: list, protected: set) -> list:
    """
    Zwraca liste indeksow tokenow konczcych zdania lub punkty wyliczenia.

    Reguly:
    1. Dwukropek przed wyliczeniem:  "mowa w : 1 )"  -> granica na ":"
    2. ; lub , przed kolejnym punktem: "sadow ; 2 )"  -> granica na ";"
    3. ; lub , na koncu segmentu:     "sadow ;"       -> granica na ";"
    4. Standardowe: . ? !
    5. ) po cyfrze jako koniec zdania — tylko gdy nie jest naglowkiem punktu
    """
    ends = []
    n = len(tokens)

    for idx in range(n):
        token = tokens[idx]

        # 1. Dwukropek przed wyliczeniem
        if token == ":":
            if idx + 2 < n and tokens[idx + 1].isdigit() and tokens[idx + 2] == ")":
                ends.append(idx)
            continue

        # 2+3. Srednik lub przecinek
        if token in {",", ";"}:
            # Regula 2: za nim jest numer kolejnego punktu "cyfra )"
            if idx + 2 < n and tokens[idx + 1].isdigit() and tokens[idx + 2] == ")":
                ends.append(idx)
                continue
            # Regula 3: jest na koncu segmentu
            rest = [t for t in tokens[idx + 1:] if t.strip()]
            if not rest:
                ends.append(idx)
            continue

        # 4. Standardowe: . ? !
        if token in {".", "?", "!"}:
            if idx > 0:
                prev = tokens[idx - 1].lower()
                if prev in protected:
                    continue
                if idx + 1 < n and (
                    tokens[idx + 1].isdigit()
                    or tokens[idx + 1].lower() in protected
                ):
                    continue
            ends.append(idx)
            continue

        # 5. ) po cyfrze
        if token == ")" and idx > 0 and tokens[idx - 1].isdigit():
            # Ochrona: "X : 1 )" lub "X ; 2 )" — nawias nalezy do numeru punktu wyliczenia
            # Sprawdzamy czy miedzy ostatnim delimiterem (: ; ,) a nawiasem jest tylko cyfra
            preceding_delims = [i for i, t in enumerate(tokens[:idx - 1]) if t in {":", ";", ","}]
            if preceding_delims:
                last_delim_idx = preceding_delims[-1]
                between = [t for t in tokens[last_delim_idx + 1:idx] if t.strip()]
                if len(between) <= 1:
                    continue  # to nawias punktu wyliczenia, nie koniec zdania
            # Numer na poczatku segmentu bez kontekstu wyliczenia
            if tokens[idx - 1].isdigit() and (idx < 5 or len(tokens) < 8):
                continue
            ends.append(idx)

    return ends


def find_sentence_end(tokens: list, protected: set) -> int:
    """Kompatybilnosc wsteczna."""
    ends = find_sentence_ends(tokens, protected)
    return ends[-1] if ends else -1


# ── Glowna logika ─────────────────────────────────────────────────────────────

def process_files(files: list, window_size: int = 5, debug: bool = False) -> list:
    """
    Etykietuje tokeny jako granice zdan (SBD).
    Zwraca liste slownikow {"tokens": [...], "sbd_labels": [...]}.
    """
    all_samples = []

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = re.sub(r"\s+", " ", f.read()).strip()

            content   = remove_headers(raw)
            segments  = proactive_split(content)
            sentences = clean_sentences(segments)
            sentences = merge_article_headers(sentences)

            if debug:
                short = [s for s in sentences if len(s.strip()) < 25]
                if short:
                    print(f"\n[DEBUG] {os.path.basename(file_path)}:")
                    for s in short[:5]:
                        print(f"  '{s}'  | HEADER_RE: {bool(HEADER_RE.match(s.strip()))}")

            for i in range(0, len(sentences), window_size):
                window          = sentences[i : i + window_size]
                combined_tokens = []
                combined_labels = []

                for sent in window:
                    tokens = tokenize(sent)
                    if not tokens:
                        continue

                    labels      = [0] * len(tokens)
                    end_indices = find_sentence_ends(tokens, PROTECTED_TERMS)

                    if end_indices:
                        for ei in end_indices:
                            labels[ei] = 1
                    else:
                        last = tokens[-1].lower()
                        if not tokens[-1].isdigit() and last not in PROTECTED_TERMS:
                            labels[-1] = 1

                    combined_tokens.extend(tokens)
                    combined_labels.extend(labels)

                n = len(combined_tokens)
                if 10 < n < 450:
                    all_samples.append({
                        "tokens":     combined_tokens,
                        "sbd_labels": combined_labels,
                    })

        except Exception as e:
            print(f"Blad w pliku {file_path}: {e}")

    return all_samples


# ── Statystyki ────────────────────────────────────────────────────────────────

def print_stats(name: str, data: list) -> None:
    total_tokens = sum(len(s["tokens"]) for s in data)
    total_ends   = sum(sum(s["sbd_labels"]) for s in data)
    ratio        = total_ends / total_tokens * 100 if total_tokens else 0
    print(f"  {name}: {len(data):,} okien | {total_tokens:,} tokenow | "
          f"{total_ends:,} granic zdan ({ratio:.1f}%)")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    BASE_DIR = "/Users/tkrezymo/LocalOnly/magisterka2/ai-segmentacja-zdan-prawniczych/"
    SAVE_PATH = "data/final_dataset_v10" 

    train_paths = glob.glob(os.path.join(BASE_DIR, "data/acts/2019/*_clean.txt"))
    test_paths  = glob.glob(os.path.join(BASE_DIR, "data/acts/2019/*_clean.txt"))

    print(f"Znaleziono plikow: train={len(train_paths)}, test={len(test_paths)}")
    print("Generowanie Datasetu V10...")

    train_data = process_files(train_paths, debug=False)
    test_data  = process_files(test_paths,  debug=False)

    print("\nStatystyki:")
    print_stats("Train", train_data)
    print_stats("Test",  test_data)

    ds_train = Dataset.from_list(train_data)
    ds_test  = Dataset.from_list(test_data)

    dataset_dict = DatasetDict({"train": ds_train, "test": ds_test})

    os.makedirs(SAVE_PATH, exist_ok=True)
    dataset_dict.save_to_disk(SAVE_PATH)

    print(f"\nDataset V10 zapisany w: {os.path.abspath(SAVE_PATH)}")


if __name__ == "__main__":
    main()
