"""
Pipeline: JSON gold -> train/dev/test.spacy -> model in spacy/result/ws2_sm.

Default sources:
  train -> data/exported_sentences_ls/exported_sentences_ls_merged.json
  dev   -> data/exported_sentences_ls/wyciete_zdania_raw_2025.json
  test  -> data/exported_sentences_ls/wyciete_zdania_raw_2026.json

Run from spacy/src:
  python build_and_train_model.py
  python build_and_train_model.py --skip-train   # conversion to .spacy only
  python build_and_train_model.py --vectors sm --window-size 2

If you run with the system Python (e.g. Anaconda), the script will switch
to the project venv (../../venv/Scripts/python.exe).
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
    """Anaconda/system Python lacks a compatible spacy — use the repo venv."""
    repo_root = Path(__file__).resolve().parents[2]
    venv_python = repo_root / "venv" / "Scripts" / "python.exe"
    if not venv_python.is_file():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    print(f"[*] Redirecting to project venv: {venv_python}", flush=True)
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_reexec_with_project_venv()

from act_text_utils import LineEndingMode
from convert_json_to_spacy import act_year_dirs, convert_train_dev
from run_gold_experiments import _pretrained_vectors_for

SCRIPT_DIR = Path(__file__).resolve().parent
SPACY_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SPACY_ROOT.parent

DATA_DIR = SPACY_ROOT / "data" / "exported_sentences_ls"
CONFIG_DIR = SPACY_ROOT / "spacy_config"
RESULT_DIR = SPACY_ROOT / "result" / "ws3_md"

DEFAULT_TRAIN_JSON = DATA_DIR / "exported_sentences_ls_merged.json"
DEFAULT_DEV_JSON = DATA_DIR / "wyciete_zdania_raw_2025.json"
DEFAULT_TEST_JSON = DATA_DIR / "wyciete_zdania_raw_2026.json"

TRAIN_ACT_YEARS = ("2016","2018", "2019", "2020", "2021")
DEV_ACT_YEARS = ("2025",)
TEST_ACT_YEARS = ("2026",)

DEFAULT_VECTORS = "pl_core_news_md"
DEFAULT_WINDOW_SIZE = 2
DEFAULT_LINE_ENDING_MODE: LineEndingMode = "space"


def _check_alignment(debug_json: Path, label: str) -> tuple[int, int]:
    data = json.loads(debug_json.read_text(encoding="utf-8"))
    ok = sum(len(d["successfully_aligned_sentences"]) for d in data)
    failed = sum(len(d["failed_alignments_chars"]) for d in data)
    print(f"  {label}: {len(data)} documents, aligned={ok}, failed={failed}", flush=True)
    return ok, failed


def convert_datasets(
    train_json: Path,
    dev_json: Path,
    test_json: Path,
    *,
    config_dir: Path = CONFIG_DIR,
    line_ending_mode: LineEndingMode = DEFAULT_LINE_ENDING_MODE,
) -> None:
    for path, name in (
        (train_json, "train"),
        (dev_json, "dev"),
        (test_json, "test"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} JSON: {path}")

    print("[1/3] Converting JSON -> train.spacy / dev.spacy / test.spacy", flush=True)
    print(f"  line_ending: {line_ending_mode}", flush=True)
    print(f"  train: {train_json}", flush=True)
    print(f"  dev:   {dev_json}", flush=True)
    print(f"  test:  {test_json}", flush=True)

    convert_train_dev(
        train_json,
        dev_json,
        train_text_dirs=act_year_dirs(REPO_ROOT, TRAIN_ACT_YEARS),
        dev_text_dirs=act_year_dirs(REPO_ROOT, DEV_ACT_YEARS),
        output_dir=config_dir,
        test_json=test_json,
        test_text_dirs=act_year_dirs(REPO_ROOT, TEST_ACT_YEARS),
        inicjaly=False,  # legal abbreviations only (art., ust., ...); not D., Sz., ...
        line_ending_mode=line_ending_mode,
    )

    train_failed = _check_alignment(config_dir / "debug_spacy_alignment_train.json", "train")
    dev_failed = _check_alignment(config_dir / "debug_spacy_alignment_dev.json", "dev")
    test_failed = _check_alignment(config_dir / "debug_spacy_alignment_test.json", "test")

    if train_failed[1] or dev_failed[1] or test_failed[1]:
        raise RuntimeError(
            "Alignment errors after conversion — check debug_spacy_alignment_*.json "
            "(often wrong _clean.txt file or act_2025_1 vs act_2025_10)."
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
            f"Missing train.spacy or dev.spacy in {config_dir}. Run conversion first."
        )

    print("[2/3] Preparing training config", flush=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = config_dir / "config.cfg"
    if not base_cfg.is_file():
        raise FileNotFoundError(f"Missing config.cfg: {base_cfg}")

    text = base_cfg.read_text(encoding="utf-8")
    text = re.sub(r'^vectors = ".*"$', f'vectors = "{vectors}"', text, count=1, flags=re.M)
    text = re.sub(
        r"^window_size = \d+$",
        f"window_size = {window_size}",
        text,
        count=1,
        flags=re.M,
    )
    use_pretrained = _pretrained_vectors_for(vectors)
    text = re.sub(
        r"^pretrained_vectors = \S+$",
        f"pretrained_vectors = {'true' if use_pretrained else 'false'}",
        text,
        count=1,
        flags=re.M,
    )
    (result_dir / "config.cfg").write_text(text, encoding="utf-8")

    print("[3/3] Training model (early stopping on dev 2025)", flush=True)
    print(
        f"  vectors={vectors}, window_size={window_size}, "
        f"pretrained_vectors={use_pretrained}",
        flush=True,
    )
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
            f"    dev F1={perf.get('sents_f', 0) * 100:.2f}% "
            f"P={perf.get('sents_p', 0) * 100:.2f}% "
            f"R={perf.get('sents_r', 0) * 100:.2f}%",
            flush=True,
        )
        print(
            f"    test.spacy ({DEFAULT_TEST_JSON.name}) ready for evaluation with a separate script",
            flush=True,
        )
    return perf


def run_pipeline(
    *,
    train_json: Path = DEFAULT_TRAIN_JSON,
    dev_json: Path = DEFAULT_DEV_JSON,
    test_json: Path = DEFAULT_TEST_JSON,
    result_dir: Path = RESULT_DIR,
    vectors: str = DEFAULT_VECTORS,
    window_size: int = DEFAULT_WINDOW_SIZE,
    line_ending_mode: LineEndingMode = DEFAULT_LINE_ENDING_MODE,
    skip_train: bool = False,
) -> None:
    convert_datasets(
        train_json,
        dev_json,
        test_json,
        line_ending_mode=line_ending_mode,
    )
    if not skip_train:
        train_model(
            result_dir=result_dir,
            vectors=vectors,
            window_size=window_size,
        )
    else:
        print("\n(--skip-train: training skipped)", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert merged/2025/2026 JSON to .spacy and train the ws2_sm model."
    )
    parser.add_argument(
        "--train-json",
        type=Path,
        default=DEFAULT_TRAIN_JSON,
        help="Train JSON (default: exported_sentences_ls_merged.json)",
    )
    parser.add_argument(
        "--dev-json",
        type=Path,
        default=DEFAULT_DEV_JSON,
        help="Dev JSON (default: wyciete_zdania_raw_2025.json)",
    )
    parser.add_argument(
        "--test-json",
        type=Path,
        default=DEFAULT_TEST_JSON,
        help="Test JSON (default: wyciete_zdania_raw_2026.json)",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=RESULT_DIR,
        help="Model output directory (default: spacy/result/ws2_sm)",
    )
    parser.add_argument("--vectors", default=DEFAULT_VECTORS)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only convert to train.spacy / dev.spacy / test.spacy",
    )
    parser.add_argument(
        "--line-ending-mode",
        choices=("lf", "space"),
        default=DEFAULT_LINE_ENDING_MODE,
        help="lf: \\r\\n -> \\n; space: \\r\\n -> space (default: space)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_pipeline(
        train_json=args.train_json.resolve(),
        dev_json=args.dev_json.resolve(),
        test_json=args.test_json.resolve(),
        result_dir=args.result_dir.resolve(),
        vectors=args.vectors,
        window_size=args.window_size,
        line_ending_mode=args.line_ending_mode,
        skip_train=args.skip_train,
    )


if __name__ == "__main__":
    main()
