# Segmentacja zdań w tekstach prawniczych (spaCy senter)

## Spis treści

1. [Model: `ws2_md_final_inicjaly`](#model-ws2_md_final_inicjaly)
2. [Uruchamianie modelu (inferencja)](#uruchamianie-modelu-inferencja)
3. [Ewaluacja](#ewaluacja)
4. [Pipeline od adnotacji do treningu](#pipeline-od-adnotacji-do-treningu)
   - [Krok 1: `convert_ls_to_json.py`](#krok-1-convert_ls_to_jsonpy)
   - [Krok 2: `merge_exported_sentences.py`](#krok-2-merge_exported_sentencespy)
   - [Krok 3: `build_and_train_model.py`](#krok-3-build_and_train_modelpy)
   - [Moduł: `convert_json_to_spacy.py`](#moduł-convert_json_to_spacypy)
5. [Eksperymenty: `run_gold_experiments.py`](#eksperymenty-run_gold_experimentspy)
6. [Struktura katalogów](#struktura-katalogów)
7. [Skrypty w `spacy/src/`](#skrypty-w-spacysrc)
8. [Pliki w `spacy/data/`](#pliki-w-spacydata)
9. [Wymagania i środowisko](#wymagania-i-środowisko)

---

## Model: `ws2_md_final_inicjaly`

Główny wytrenowany model znajduje się w:

```
spacy/result/ws2_md_final_inicjaly/model-best/
```

Jest to segmentator polskich aktów prawniczych na zdania oparty o komponent **spaCy `senter`**, wytrenowany wyłącznie na danych olabelowanych ręcznie.

| Parametr | Wartość |
|----------|---------|
| Architektura | `HashEmbedCNN.v2` + `Tagger.v2` (senter) |
| Wektory | `pl_core_news_md` (`pretrained_vectors = true`) |
| `window_size` | 2 |
| `width` (tok2vec) | 32 |
| `embed_size` | 2000 |
| Pipeline | tylko `["senter"]` |
| Tokenizer | skróty prawnicze (`art.`, `ust.`, `Dz. U.`, …) + **inicjały** (`D.`, `Sz.`, …) |
| Parametry trenowalne | ~186 tys. (wektory md ~6 mln — zamrożone, z `pl_core_news_md`) |

**Podział danych:**

| Zbiór | Lata aktów | Plik JSON | Plik `.spacy` |
|-------|------------|-----------|---------------|
| Train | 2016, 2018–2021 | `exported_sentences_ls_merged.json` | `spacy_config/train.spacy` |
| Dev | 2025 | `wyciete_zdania_raw_2025.json` | `spacy_config/dev.spacy` |
| Test | 2026 | `wyciete_zdania_raw_2026.json` | `spacy_config/test.spacy` |

**Metryki:**

| Zbiór | SENT P | SENT R | SENT F |
|-------|--------|--------|--------|
| Dev (meta modelu) | ~91,0% | ~89,9% | ~90,4% |
| Test (`spacy evaluate`) | ~92,9% | ~91,4% | ~92,1% |


---

## Uruchamianie modelu

Wszystkie komendy powinny być uruchamiane z katalogu `spacy/src/`, najlepiej przez venv. Wymagane biblioteki znajdują się w `spacy/requirements.txt`:

```powershell
cd spacy\src
..\..\venv\Scripts\python.exe segment_file.py `
  --model ../result/ws2_md_final_inicjaly/model-best `
  --input "../../legal-text-downloader/data/acts_for_gold/2026/act_2026_732_..._clean.txt"
```

Wynik: plik `*__segmented.txt` obok pliku wejściowego, format:

```
[0] pierwsze zdanie
[1] drugie zdanie
...
```

**Opcje `segment_file.py`:**

| Flaga | Opis |
|-------|------|
| `--model` | Katalog `model-best` (domyślnie `result/ws2_sm/model-best`) |
| `--input` | Plik `_clean.txt` do podziału |
| `--output` | Ścieżka wyjścia (domyślnie: `__segmented.txt` obok wejścia) |
| `--line-ending-mode` | `space` (domyślnie) lub `lf`|


## Ewaluacja

**Metryki na dev/test:**

```powershell
cd spacy\src
..\..\venv\Scripts\python.exe -m spacy evaluate `
  ../result/ws2_md_final_inicjaly/model-best `
  ../spacy_config/test.spacy
```

---

## Pipeline od adnotacji do treningu

```
Label Studio (gold_YYYY.json)
        │
        ▼
convert_ls_to_json.py          →  wyciete_zdania_raw_YYYY.json
        │
        ▼
merge_exported_sentences.py    →  exported_sentences_ls_merged.json  (train)
        │
        ▼
build_and_train_model.py       →  train.spacy / dev.spacy / test.spacy
        │                          + model w result/
        ▼
segment_file.py / spacy evaluate
```

### Krok 1: `convert_ls_to_json.py`

Eksport adnotacji z Label Studio do formatu JSON używanego w projekcie.

- **Wejście:** `spacy/data/gold_YYYY.json` (eksport Label Studio)
- **Teksty:** pliki `act_YYYY_N_*_clean.txt`
- **Wyjście:** `data/exported_sentences_ls/wyciete_zdania_raw_YYYY.json`

Każdy dokument w JSON ma postać:

```json
{
  "act_id": "act_2026_641",
  "sentences": [
    {"start_offset": 0, "end_offset": 71, "text": "..."},
    ...
  ]
}
```

Offsety są liczone względem tekstu po normalizacji końców linii (`lf` lub `space`).

**Typowe komendy:**

```powershell
# Wszystkie lata naraz
python convert_ls_to_json.py --export-all-years --mode lf

# Pojedynczy rok
python convert_ls_to_json.py --ls-json ../data/gold_2016.json `
  --txt-dir ../../legal-text-downloader/data/acts_for_gold/2016/ `
  --output ../data/exported_sentences_ls/wyciete_zdania_raw_2016.json `
  --mode lf

# Normalizacja istniejących plików (CRLF, offsety)
python convert_ls_to_json.py --normalize-existing --mode lf
```

| Flaga | Opis |
|-------|------|
| `--export-all-years` | Eksport `gold_YYYY.json` → `wyciete_zdania_raw_YYYY.json` dla wszystkich lat |
| `--mode lf\|space` | `lf`: `\r\n` → `\n`; `space`: `\r\n` → spacja |
| `--normalize-existing` | Popraw istniejące `wyciete_zdania_raw_*.json` w `exported_sentences_ls/` |

### Krok 2: `merge_exported_sentences.py`

Scala pliki roczne w jeden zbiór treningowy.

- **Wejście:** `wyciete_zdania_raw_*.json` (bez `_2025` i `_2026` — te idą na dev/test)
- **Wyjście:** `data/exported_sentences_ls/exported_sentences_ls_merged.json`

```powershell
python merge_exported_sentences.py
```

### Krok 3: `build_and_train_model.py`

Główny skrypt pipeline'u: konwersja JSON → `.spacy` + trening modelu.

```powershell
# Pełny pipeline (konwersja + trening)
python build_and_train_model.py

# Tylko konwersja do .spacy (bez treningu)
python build_and_train_model.py --skip-train

# Zmiana wektorów / okna kontekstu
python build_and_train_model.py --vectors md --window-size 2 --result-dir ../result/moj_model
```

**Domyślne źródła:**

| Zbiór | Plik JSON |
|-------|-----------|
| Train | `exported_sentences_ls_merged.json` |
| Dev | `wyciete_zdania_raw_2025.json` |
| Test | `wyciete_zdania_raw_2026.json` |

**Domyślne wyjścia:** `spacy_config/train.spacy`, `dev.spacy`, `test.spacy` oraz model w `result/ws2_md_inicjaly_final/`

| Flaga | Opis |
|-------|------|
| `--skip-train` | Tylko konwersja JSON → `.spacy` |
| `--vectors sm\|md\|lg` | Pakiet wektorów spaCy (domyślnie `md`) |
| `--window-size N` | Okno kontekstu HashEmbedCNN (domyślnie 2) |
| `--line-ending-mode lf\|space` | Tryb końców linii (domyślnie `space`) |
| `--result-dir` | Katalog wyjściowy modelu |


### Moduł: `convert_json_to_spacy.py`

Moduł biblioteczny używany przez `build_and_train_model.py`. Można go też uruchomić bezpośrednio.

Odpowiada za:

- tokenizer treningowy (`spacy.blank("pl")` + wyjątki ORTH dla skrótów prawniczych i opcjonalnie inicjałów),
- mapowanie offsetów z JSON na tokeny (`is_sent_start`),
- zapis `DocBin` → `.spacy`,

---

## Eksperymenty: `run_gold_experiments.py`

Siatka treningów na danych gold: **`pl_core_news_sm` / `md` / `lg` × `window_size`**.

Wyniki trafiają do `spacy_config/experiments_gold/` (lub wariantów `experiments_gold_space/` itd., zależnie od flag).

```powershell
# Pełna siatka (domyślnie)
python run_gold_experiments.py

# Wybrane okna i wektory
python run_gold_experiments.py --windows 1 3 5 --vectors sm lg

# Szybki test (mniej kroków)
python run_gold_experiments.py --max-steps 2000

# Podgląd planu bez treningu
python run_gold_experiments.py --dry-run
```

Przydatne flagi: `--inicjaly`, `--line-ending-mode lf|space`, `--skip-existing`, `--reconvert`.

Skrypt `build_and_train_model.py` importuje z niego helper `_pretrained_vectors_for()` (logika `pretrained_vectors` dla sm vs md/lg).

---

## Struktura katalogów

```
spacy/
├── README.md                 ← ten plik
├── data/                     ← adnotacje JSON i eksporty Label Studio
├── src/                      ← skrypty Python
├── spacy_config/
│   ├── config.cfg            ← szablon konfiguracji treningu spaCy
│   ├── train.spacy           ← korpus treningowy
│   ├── dev.spacy             ← korpus deweloperski (2025)
│   ├── test.spacy            ← korpus testowy (2026)
│   └── debug_spacy_alignment_*.json
└── result/
    └── ws2_md_final_inicjaly/
        ├── model-best/       ← wytrenowany model
        ├── error_analysis/   ← błędy na dev
        └── error_analysis_test/
```

---

## Skrypty w `spacy/src/`

W katalogu znajdują się **6 aktywnych plików Python** (stan na czerwiec 2026):

| Skrypt | Rola |
|--------|------|
| `convert_ls_to_json.py` | Eksport Label Studio → `wyciete_zdania_raw_YYYY.json` |
| `merge_exported_sentences.py` | Scalanie rocznych JSON-ów w plik treningowy |
| `convert_json_to_spacy.py` | Konwersja JSON → `.spacy` (tokenizer, alignment) |
| `build_and_train_model.py` | Pipeline: JSON → `.spacy` → trening modelu |
| `run_gold_experiments.py` | Siatka eksperymentów (wektory × window_size) |
| `segment_file.py` | Segmentacja pojedynczego pliku `_clean.txt` → `__segmented.txt` |


---

## Pliki w `spacy/data/`

### Eksporty Label Studio (`gold_*.json`)

Surowe adnotacje wyeksportowane z Label Studio — wejście do `convert_ls_to_json.py`.

| Plik | Opis |
|------|------|
| `gold_2016.json` | Adnotacje aktów z 2016 |
| `gold_2018.json` | Adnotacje aktów z 2018 |
| `gold_2019.json` | Adnotacje aktów z 2019 |
| `gold_2020.json` | Adnotacje aktów z 2020 |
| `gold_2021.json` | Adnotacje aktów z 2021 |
| `gold_2025.json` | Adnotacje aktów z 2025 (dev) |
| `gold_2026.json` | Adnotacje aktów z 2026 (test) |
| `gold.zip`, `gold_final.zip` | Archiwa kopii zapasowych adnotacji |

### Przetworzone JSON (`data/exported_sentences_ls/`)

| Plik / katalog | Opis |
|----------------|------|
| `wyciete_zdania_raw_2016.json` | Zdania + offsety — rok 2016 |
| `wyciete_zdania_raw_2018.json` | … 2018 |
| `wyciete_zdania_raw_2019.json` | … 2019 |
| `wyciete_zdania_raw_2020.json` | … 2020 |
| `wyciete_zdania_raw_2021.json` | … 2021 |
| `wyciete_zdania_raw_2025.json` | … 2025 (dev) |
| `wyciete_zdania_raw_2026.json` | … 2026 (test) |
| `exported_sentences_ls_merged.json` | Scalony train (2016, 2018–2021) |
| `gold_2025.json` | Kopia / wariant eksportu LS dla 2025 |
| `validation_report_*.json` | Raporty walidacji offsetów po eksporcie |
| `backup z enterem/` | Kopia zapasowa eksportów z `\n` zamiast spacji w tekście zdań |

### Inne

| Plik | Opis |
|------|------|
| `results/first/wyniki_dla_801sentences_2020.md` | Wczesne wyniki eksperymentów (2020) |

---

## Wymagania i środowisko

- Python 3.11+ (venv projektu: `../../venv/`)
- spaCy ≥ 3.8
- Modele językowe: `pl_core_news_md` (i opcjonalnie `sm`, `lg`)
- Teksty aktów: `legal-text-downloader/data/acts_for_gold/YYYY/act_*_clean.txt`

**Instalacja modelu językowego (jeśli brak):**

```powershell
python -m spacy download pl_core_news_md
```

**Pełny pipeline od zera (skrót):**

```powershell
cd spacy\src

python convert_ls_to_json.py --export-all-years --mode lf
python merge_exported_sentences.py
python build_and_train_model.py --line-ending-mode lf --result-dir ../result/ws2_md_final_inicjaly
```

Po treningu model znajdziesz w `result/ws2_md_final_inicjaly/model-best/`.
