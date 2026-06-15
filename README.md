# Automatyczna segmentacja polskich tekstów prawnych na zdania

**Praca dyplomowa (projektowa)**

| | |
|---|---|
| **Temat (PL)** | Automat dzielący polski tekst prawny na zdania z wykorzystaniem mechanizmów uczenia maszynowego |

---

## Spis treści

1. [O projekcie](#o-projekcie)
2. [Źródła danych](#źródła-danych)
3. [Środowisko](#środowisko)

---

## O projekcie

Polskie teksty prawne — ustawy, rozporządzenia, obwieszczenia czy orzeczenia — są pisane specyficznym językiem prawniczym: gęsto występują skróty (`art.`, `ust.`, `Dz. U.`), numeracja paragrafów i punktów, inicjały w podpisach oraz układ typowy dla publikacji w Dzienniku Ustaw. Ogólne narzędzia do dzielenia tekstu na zdania (spaCy, NLTK, parsery językowe) na takich materiałach **popełniają systematyczne błędy** — łączą nagłówki z treścią, rozdzielają listy w niewłaściwych miejscach albo traktują skróty jak koniec zdania.
---


### `spacy/` — model spaCy

Katalog **`spacy/`** grupuje cały pipeline oparty o **spaCy `senter`**:
 
Szczegóły, komendy i opis skryptów: **[`spacy/README.md`](spacy/README.md)**.

### `TBD/` — transformery (BERT i inne)

Katalog **`TBD/`** (to be done / w przygotowaniu) jest przeznaczony na **alternatywne podejścia oparte o modele transformera** — m.in. fine-tuning BERT (polskich wariantów językowych) do zadania wykrywania granic zdań w tekstach prawniczych. Na ten moment główny, dopracowany pipeline znajduje się w `spacy/`; implementacja w `TBD/` uzupełni porównanie metod z punktu (2) pracy dyplomowej.

### `legal-text-downloader/`

Skrypt `get_legal_text.py` pobiera akty prawne i zapisuje wersje oczyszczone (`*_clean.txt`) w `data/acts_for_gold/<rok>/`. Te pliki są źródłem tekstu dla Label Studio i dla offsetów w JSON.

### `segmentation_tests/`

Pliki segmentacji wykorzystane w pracy z podziałem na rozdziały — porównania metod (NLTK, spaCy sm/md/lg, Stanza) oraz wyniki własnego modelu na przykładowych aktach.

---

## Źródła danych

| Źródło | Opis |
|--------|------|
| [Dziennik Ustaw](https://isap.sejm.gov.pl/) | Akty prawne 
| Label Studio | Ręczna adnotacja granic zdań → `spacy/data/gold_*.json` |


---

## Środowisko

- **Python 3.11+**
- Kluczowe pakiety: `spacy` (≥ 3.8), `pl_core_news_md`, narzędzia do adnotacji i pobierania tekstów (patrz `requirements.txt` w podkatalogach)

---
