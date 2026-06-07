"""
Kopiuje pliki *_clean.txt potrzebne do gold train/dev do acts_for_gold/ (~2 MB).
Pelny korpus (legal-text-downloader/data/acts/) jest w .gitignore — mozna go pobrac
ponownie skryptem get_legal_text.py.

Uruchomienie (z katalogu spacy/src):
  python export_gold_acts.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SOURCE_ACTS = REPO_ROOT / "legal-text-downloader" / "data" / "acts"
TARGET_ACTS = REPO_ROOT / "legal-text-downloader" / "data" / "acts_for_gold"

GOLD_JSONS = [
    SCRIPT_DIR / "../data/exported_sentences_ls/exported_sentences_ls_merged.json",
    SCRIPT_DIR / "../data/exported_sentences_ls/wyciete_zdania_raw_2026.json",
]


def collect_act_ids() -> set[str]:
    ids: set[str] = set()
    for path in GOLD_JSONS:
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data:
            ids.add(item["act_id"])
    return ids


def export_acts() -> None:
    act_ids = collect_act_ids()
    copied = 0
    missing: list[str] = []

    for act_id in sorted(act_ids):
        year = act_id.split("_")[1]
        src_dir = SOURCE_ACTS / year
        matches = list(src_dir.glob(f"{act_id}_*_clean.txt"))
        if not matches:
            missing.append(act_id)
            continue
        dst_dir = TARGET_ACTS / year
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(matches[0], dst_dir / matches[0].name)
        copied += 1

    print(f"[*] Skopiowano {copied} plikow do {TARGET_ACTS}")
    if missing:
        raise SystemExit(f"Brak tekstow zrodlowych dla: {missing[:5]} ... ({len(missing)} total)")


if __name__ == "__main__":
    export_acts()
