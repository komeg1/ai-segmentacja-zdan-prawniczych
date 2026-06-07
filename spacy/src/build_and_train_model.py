"""
Pipeline: JSON gold -> train/dev.spacy -> model w spacy/result/ws2_sm.

Domyslne zrodla:
  train  -> data/exported_sentences_ls/exported_sentences_ls_merged.json
  dev    -> data/exported_sentences_ls/wyciete_zdania_raw_2026.json

Uruchomienie (z katalogu spacy/src):
  python build_and_train_model.py
  python build_and_train_model.py --skip-train   # tylko konwersja .spacy
  python build_and_train_model.py --vectors sm --window-size 2

Jesli uruchomisz systemowym Pythonem (np. Anaconda), skrypt sam przełączy
się na venv projektu (../../venv/Scripts/python.exe).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def _reexec_with_project_venv() -> None:
    """Anaconda/system Python nie ma kompatybilnego spacy — uzyj venv repo."""
    repo_root = Path(__file__).resolve().parents[2]
    venv_python = repo_root / "venv" / "Scripts" / "python.exe"
    if not venv_python.is_file():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    print(f"[*] Przekierowanie na venv projektu: {venv_python}", flush=True)
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_reexec_with_project_venv()

from convert_json_to_spacy import convert_train_dev, resolve_acts_dir

SCRIPT_DIR = Path(__file__).resolve().parent
SPACY_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SPACY_ROOT.parent

DATA_DIR = SPACY_ROOT / "data" / "exported_sentences_ls"
CONFIG_DIR = SPACY_ROOT / "spacy_config"
ACTS_DIR = resolve_acts_dir(REPO_ROOT)
RESULT_DIR = SPACY_ROOT / "result" / "ws2_sm"

DEFAULT_TRAIN_JSON = DATA_DIR / "exported_sentences_ls_merged.json"
DEFAULT_DEV_JSON = DATA_DIR / "wyciete_zdania_raw_2026.json"

TRAIN_ACT_YEARS = ("2018", "2020", "2021", "2025")
DEV_ACT_YEARS = ("2026",)

DEFAULT_VECTORS = "pl_core_news_sm"
DEFAULT_WINDOW_SIZE = 2


def _check_alignment(debug_json: Path, label: str) -> tuple[int, int]:
    data = json.loads(debug_json.read_text(encoding="utf-8"))
    ok = sum(len(d["successfully_aligned_sentences"]) for d in data)
    failed = sum(len(d["failed_alignments_chars"]) for d in data)
    print(f"  {label}: {len(data)} dokumentow, aligned={ok}, failed={failed}", flush=True)
    return ok, failed


def convert_datasets(
    train_json: Path,
    dev_json: Path,
    *,
    config_dir: Path = CONFIG_DIR,
) -> None:
    if not train_json.is_file():
        raise FileNotFoundError(f"Brak train JSON: {train_json}")
    if not dev_json.is_file():
        raise FileNotFoundError(f"Brak dev JSON: {dev_json}")

    print("[1/3] Konwersja JSON -> train.spacy / dev.spacy", flush=True)
    print(f"  train: {train_json}", flush=True)
    print(f"  dev:   {dev_json}", flush=True)

    convert_train_dev(
        train_json,
        dev_json,
        train_text_dirs=[ACTS_DIR / y for y in TRAIN_ACT_YEARS],
        dev_text_dirs=[ACTS_DIR / y for y in DEV_ACT_YEARS],
        output_dir=config_dir,
    )

    train_failed = _check_alignment(config_dir / "debug_spacy_alignment_train.json", "train")
    dev_failed = _check_alignment(config_dir / "debug_spacy_alignment_dev.json", "dev")

    if train_failed[1] or dev_failed[1]:
        raise RuntimeError(
            "Alignment errors po konwersji — sprawdz debug_spacy_alignment_*.json "
            "(czesto zly plik _clean.txt lub act_2025_1 vs act_2025_10)."
        )


def train_model(
    *,
    result_dir: Path = RESULT_DIR,
    config_dir: Path = CONFIG_DIR,
    vectors: str = DEFAULT_VECTORS,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> dict:
    train_spacy = config_dir / "train.spacy"
    dev_spacy = config_dir / "dev.spacy"
    if not train_spacy.is_file() or not dev_spacy.is_file():
        raise FileNotFoundError(
            f"Brak train.spacy lub dev.spacy w {config_dir}. Uruchom konwersje najpierw."
        )

    print("[2/3] Przygotowanie config treningu", flush=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = config_dir / "config.cfg"
    if not base_cfg.is_file():
        raise FileNotFoundError(f"Brak config.cfg: {base_cfg}")

    text = base_cfg.read_text(encoding="utf-8")
    text = re.sub(r'^vectors = ".*"$', f'vectors = "{vectors}"', text, count=1, flags=re.M)
    text = re.sub(
        r"^window_size = \d+$",
        f"window_size = {window_size}",
        text,
        count=1,
        flags=re.M,
    )
    (result_dir / "config.cfg").write_text(text, encoding="utf-8")

    print("[3/3] Trening modelu", flush=True)
    print(f"  vectors={vectors}, window_size={window_size}", flush=True)
    print(f"  output={result_dir}", flush=True)

    cmd = [
        sys.executable,
        "-m",
        "spacy",
        "train",
        str(result_dir / "config.cfg"),
        "--output",
        str(result_dir),
        "--paths.train",
        str(train_spacy.resolve()),
        "--paths.dev",
        str(dev_spacy.resolve()),
    ]
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=config_dir, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    meta_path = result_dir / "model-best" / "meta.json"
    perf: dict = {}
    if meta_path.is_file():
        perf = json.loads(meta_path.read_text(encoding="utf-8")).get("performance", {})
        print(
            f"\n[+] Model: {result_dir / 'model-best'}",
            flush=True,
        )
        print(
            f"    F1={perf.get('sents_f', 0) * 100:.2f}% "
            f"P={perf.get('sents_p', 0) * 100:.2f}% "
            f"R={perf.get('sents_r', 0) * 100:.2f}%",
            flush=True,
        )
    return perf


def run_pipeline(
    *,
    train_json: Path = DEFAULT_TRAIN_JSON,
    dev_json: Path = DEFAULT_DEV_JSON,
    result_dir: Path = RESULT_DIR,
    vectors: str = DEFAULT_VECTORS,
    window_size: int = DEFAULT_WINDOW_SIZE,
    skip_train: bool = False,
) -> None:
    convert_datasets(train_json, dev_json)
    if not skip_train:
        train_model(
            result_dir=result_dir,
            vectors=vectors,
            window_size=window_size,
        )
    else:
        print("\n(--skip-train: pominięto trening)", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Konwertuj merged/2026 JSON na .spacy i wytrenuj model ws2_sm."
    )
    parser.add_argument(
        "--train-json",
        type=Path,
        default=DEFAULT_TRAIN_JSON,
        help="JSON train (domyslnie exported_sentences_ls_merged.json)",
    )
    parser.add_argument(
        "--dev-json",
        type=Path,
        default=DEFAULT_DEV_JSON,
        help="JSON dev (domyslnie wyciete_zdania_raw_2026.json)",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=RESULT_DIR,
        help="Katalog wyjsciowy modelu (domyslnie spacy/result/ws2_sm)",
    )
    parser.add_argument("--vectors", default=DEFAULT_VECTORS)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Tylko konwersja do train.spacy / dev.spacy",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_pipeline(
        train_json=args.train_json.resolve(),
        dev_json=args.dev_json.resolve(),
        result_dir=args.result_dir.resolve(),
        vectors=args.vectors,
        window_size=args.window_size,
        skip_train=args.skip_train,
    )


if __name__ == "__main__":
    main()
