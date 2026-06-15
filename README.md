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

Praca dotyczy **segmentacji zdań** w polskich aktach prawnych. Ogólne narzędzia (spaCy, NLTK) źle radzą sobie ze skrótami, numeracją i układem Dziennika Ustaw — stąd własny model douczony na ręcznie adnotowanym korpusie gold.

| Katalog | Zawartość |
|---------|-----------|
| [`spacy/`](spacy/) | Pipeline spaCy `senter`, korpus gold, wytrenowany model — szczegóły w [`spacy/README.md`](spacy/README.md) |
| `TBD/` | (w przygotowaniu) BERT i inne podejścia transformera |
| [`legal-text-downloader/`](legal-text-downloader/) | Pobieranie aktów → pliki `*_clean.txt` |
| [`segmentation_tests/`](segmentation_tests/) | Przykładowe wyniki segmentacji do rozdziałów pracy |

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
