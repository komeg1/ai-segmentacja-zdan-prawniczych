"""
Siatka eksperymentów: pl_core_news_sm / md / lg × window_size na danych GOLD.

Tylko gold (bez silver):
  train → spacy_config/train.spacy  (2977 zdań, 60 aktów: 2018+2020+2021)
  dev   → spacy_config/dev.spacy    (1976 zdań, 37 aktów: 2026)

Uruchomienie (z katalogu spacy/src):
  python run_gold_experiments.py
  python run_gold_experiments.py --windows 1 3 5 --vectors sm lg
  python run_gold_experiments.py --max-steps 2000   # szybki test
  python run_gold_experiments.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SPACY_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = SPACY_ROOT / "spacy_config"
BASE_CONFIG = CONFIG_DIR / "config.cfg"

GOLD_TRAIN = CONFIG_DIR / "train.spacy"
GOLD_DEV = CONFIG_DIR / "dev.spacy"

EXPERIMENTS_DIR = CONFIG_DIR / "experiments_gold"
SUMMARY_MD = EXPERIMENTS_DIR / "porownanie_gold_window_vectors.md"
SUMMARY_JSON = EXPERIMENTS_DIR / "porownanie_gold_window_vectors.json"

DEFAULT_VECTORS = ("pl_core_news_sm", "pl_core_news_md", "pl_core_news_lg")
DEFAULT_WINDOWS = (1, 2, 3, 4, 5)

TRAIN_DOCS = 67
TRAIN_SENTS = 3436
DEV_DOCS = 61
DEV_SENTS = 3686


@dataclass
class ExperimentResult:
    experiment_id: str
    vectors: str
    window_size: int
    sents_f: float | None
    sents_p: float | None
    sents_r: float | None
    senter_loss: float | None
    model_size_mb: float | None
    train_time_s: float
    output_dir: str
    status: str
    error: str | None = None


def _short_vectors(name: str) -> str:
    return name.replace("pl_core_news_", "")


def _experiment_id(vectors: str, window_size: int) -> str:
    return f"ws{window_size}_{_short_vectors(vectors)}"


def _write_config(vectors: str, window_size: int, path: Path) -> None:
    text = BASE_CONFIG.read_text(encoding="utf-8")
    text = re.sub(r'^vectors = ".*"$', f'vectors = "{vectors}"', text, count=1, flags=re.M)
    text = re.sub(
        r"^window_size = \d+$",
        f"window_size = {window_size}",
        text,
        count=1,
        flags=re.M,
    )
    path.write_text(text, encoding="utf-8")


def _model_size_mb(model_dir: Path) -> float | None:
    if not model_dir.is_dir():
        return None
    total = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    return round(total / 1024 / 1024, 2)


def _read_meta(model_dir: Path) -> dict:
    meta_path = model_dir / "meta.json"
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _format_table(results: list[ExperimentResult]) -> str:
    lines = [
        "# Porównanie eksperymentów GOLD — wektory × window_size",
        "",
        f"Train: {GOLD_TRAIN.relative_to(SPACY_ROOT)} ({TRAIN_SENTS} granic zdań, {TRAIN_DOCS} aktów)",
        f"Dev:   {GOLD_DEV.relative_to(SPACY_ROOT)} ({DEV_SENTS} granic zdań, {DEV_DOCS} aktów)",
        "",
        "| Eksperyment | Wektory | Window | SENTS_F | SENTS_P | SENTS_R | Model [MB] | Czas [s] | Status |",
        "|-------------|---------|--------|---------|---------|---------|------------|----------|--------|",
    ]
    for r in sorted(results, key=lambda x: (x.window_size, x.vectors)):
        f1 = f"{r.sents_f * 100:.2f}%" if r.sents_f is not None else "—"
        p = f"{r.sents_p * 100:.2f}%" if r.sents_p is not None else "—"
        rec = f"{r.sents_r * 100:.2f}%" if r.sents_r is not None else "—"
        size = f"{r.model_size_mb:.1f}" if r.model_size_mb is not None else "—"
        lines.append(
            f"| {r.experiment_id} | {_short_vectors(r.vectors)} | {r.window_size} "
            f"| {f1} | {p} | {rec} | {size} | {r.train_time_s:.0f} | {r.status} |"
        )

    ok = [r for r in results if r.sents_f is not None]
    if ok:
        best = max(ok, key=lambda r: r.sents_f or 0)
        lines.extend([
            "",
            f"*Najlepszy F1:* {best.experiment_id} — "
            f"F1={best.sents_f * 100:.2f}%, window={best.window_size}, "
            f"vectors={_short_vectors(best.vectors)}, model={best.model_size_mb} MB",
        ])
    return "\n".join(lines) + "\n"


def _run_experiment(
    vectors: str,
    window_size: int,
    *,
    max_steps: int | None,
    skip_existing: bool,
    python_exe: str,
) -> ExperimentResult:
    exp_id = _experiment_id(vectors, window_size)
    out_dir = EXPERIMENTS_DIR / exp_id
    config_path = out_dir / "config.cfg"
    model_best = out_dir / "model-best"

    if skip_existing and (model_best / "meta.json").is_file():
        meta = _read_meta(model_best)
        perf = meta.get("performance", {})
        return ExperimentResult(
            experiment_id=exp_id,
            vectors=vectors,
            window_size=window_size,
            sents_f=perf.get("sents_f"),
            sents_p=perf.get("sents_p"),
            sents_r=perf.get("sents_r"),
            senter_loss=perf.get("senter_loss"),
            model_size_mb=_model_size_mb(model_best),
            train_time_s=0.0,
            output_dir=str(out_dir),
            status="skipped (exists)",
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_config(vectors, window_size, config_path)

    cmd = [
        python_exe,
        "-m",
        "spacy",
        "train",
        str(config_path),
        "--output",
        str(out_dir),
        "--paths.train",
        str(GOLD_TRAIN.resolve()),
        "--paths.dev",
        str(GOLD_DEV.resolve()),
    ]
    if max_steps is not None:
        cmd.extend(["--training.max_steps", str(max_steps)])

    print(f"\n{'=' * 60}", flush=True)
    print(f"Eksperyment: {exp_id}", flush=True)
    print(f"  vectors={vectors}, window_size={window_size}", flush=True)
    print(f"  output={out_dir}", flush=True)
    print(" ".join(cmd), flush=True)

    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=CONFIG_DIR, check=False, capture_output=True, text=True)
        elapsed = time.perf_counter() - start

        log_path = out_dir / "train.log"
        log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")

        if proc.returncode != 0:
            return ExperimentResult(
                experiment_id=exp_id,
                vectors=vectors,
                window_size=window_size,
                sents_f=None,
                sents_p=None,
                sents_r=None,
                senter_loss=None,
                model_size_mb=None,
                train_time_s=elapsed,
                output_dir=str(out_dir),
                status="failed",
                error=(proc.stderr or proc.stdout or "")[-2000:],
            )

        meta = _read_meta(model_best)
        perf = meta.get("performance", {})
        return ExperimentResult(
            experiment_id=exp_id,
            vectors=vectors,
            window_size=window_size,
            sents_f=perf.get("sents_f"),
            sents_p=perf.get("sents_p"),
            sents_r=perf.get("sents_r"),
            senter_loss=perf.get("senter_loss"),
            model_size_mb=_model_size_mb(model_best),
            train_time_s=elapsed,
            output_dir=str(out_dir),
            status="ok",
        )
    except Exception as exc:
        return ExperimentResult(
            experiment_id=exp_id,
            vectors=vectors,
            window_size=window_size,
            sents_f=None,
            sents_p=None,
            sents_r=None,
            senter_loss=None,
            model_size_mb=None,
            train_time_s=time.perf_counter() - start,
            output_dir=str(out_dir),
            status="error",
            error=str(exc),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Siatka treningu senter na gold.")
    parser.add_argument(
        "--vectors",
        nargs="+",
        default=list(DEFAULT_VECTORS),
        help="Pakiety wektorów: sm md lg lub pełne nazwy pl_core_news_*",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=list(DEFAULT_WINDOWS),
        help="Wartości window_size do przetestowania",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Nadpisanie max_steps (domyślnie z config.cfg: 20000)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Pomiń eksperymenty z istniejącym model-best/meta.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Wypisz plan bez uruchamiania treningu",
    )
    return parser.parse_args()


def _normalize_vectors(names: list[str]) -> list[str]:
    mapping = {"sm": "pl_core_news_sm", "md": "pl_core_news_md", "lg": "pl_core_news_lg"}
    out = []
    for name in names:
        out.append(mapping.get(name, name))
    return out


def main() -> None:
    args = _parse_args()
    vectors_list = _normalize_vectors(args.vectors)
    windows = sorted(set(args.windows))

    if not BASE_CONFIG.is_file():
        raise FileNotFoundError(f"Brak config.cfg: {BASE_CONFIG}")
    if not GOLD_TRAIN.is_file() or not GOLD_DEV.is_file():
        raise FileNotFoundError(
            f"Brak danych gold:\n  train: {GOLD_TRAIN}\n  dev: {GOLD_DEV}\n"
            "Wygeneruj je skryptem convert_json_to_spacy.py"
        )

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    plan = [(v, w) for v in vectors_list for w in windows]

    print("Plan eksperymentow (tylko GOLD):", flush=True)
    print(f"  train: {GOLD_TRAIN}", flush=True)
    print(f"  dev:   {GOLD_DEV}", flush=True)
    for v, w in plan:
        print(f"  - {_experiment_id(v, w)}", flush=True)

    if args.dry_run:
        print("\n(dry-run — bez treningu)")
        return

    results: list[ExperimentResult] = []
    python_exe = sys.executable

    for vectors, window_size in plan:
        result = _run_experiment(
            vectors,
            window_size,
            max_steps=args.max_steps,
            skip_existing=args.skip_existing,
            python_exe=python_exe,
        )
        results.append(result)
        if result.sents_f is not None:
            print(
                f"[TRENING UKONCZONY] {result.experiment_id} | "
                f"F1={result.sents_f * 100:.2f}% "
                f"P={result.sents_p * 100:.2f}% R={result.sents_r * 100:.2f}% | "
                f"model={result.model_size_mb} MB | czas={result.train_time_s:.0f}s",
                flush=True,
            )
        else:
            print(f"[TRENING UKONCZONY] {result.experiment_id} | {result.status}", flush=True)
            if result.error:
                print(f"  blad: {result.error[:500]}", flush=True)

    SUMMARY_JSON.write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    table = _format_table(results)
    SUMMARY_MD.write_text(table, encoding="utf-8")

    print(f"\n{table}")
    print(f"Zapisano: {SUMMARY_MD}")
    print(f"Zapisano: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
