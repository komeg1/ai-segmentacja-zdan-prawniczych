"""
Siatka eksperymentów: ws2_sm (pl_core_news_sm, window_size=2) × width tok2vec.

Uruchomienie (z katalogu spacy/src):
  python run_gold_width_experiments.py
  python run_gold_width_experiments.py --widths 12 32 64 128
  python run_gold_width_experiments.py --dry-run
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

EXPERIMENTS_DIR = CONFIG_DIR / "experiments_gold_final"
SUMMARY_MD = EXPERIMENTS_DIR / "porownanie_width_ws2_sm.md"
SUMMARY_JSON = EXPERIMENTS_DIR / "porownanie_width_ws2_sm.json"

VECTORS = "pl_core_news_sm"
WINDOW_SIZE = 2
DEFAULT_WIDTHS = (12, 32, 64, 128)

TRAIN_DOCS = 59
TRAIN_SENTS = 3160
DEV_DOCS = 46
DEV_SENTS = 2620


@dataclass
class ExperimentResult:
    experiment_id: str
    width: int
    window_size: int
    vectors: str
    sents_f: float | None
    sents_p: float | None
    sents_r: float | None
    senter_loss: float | None
    model_size_mb: float | None
    train_time_s: float
    output_dir: str
    status: str
    error: str | None = None


def _experiment_id(width: int) -> str:
    return f"width{width}_ws2_sm"


def _write_config(width: int, path: Path) -> None:
    text = BASE_CONFIG.read_text(encoding="utf-8")
    text = re.sub(r'^vectors = ".*"$', f'vectors = "{VECTORS}"', text, count=1, flags=re.M)
    text = re.sub(
        r"^window_size = \d+$",
        f"window_size = {WINDOW_SIZE}",
        text,
        count=1,
        flags=re.M,
    )
    text = re.sub(
        r"^width = \d+$",
        f"width = {width}",
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
        "# Porownanie eksperymentow GOLD final — ws2_sm x width",
        "",
        f"Bazowa konfiguracja: vectors={VECTORS}, window_size={WINDOW_SIZE}",
        f"Train: spacy_config/train.spacy ({TRAIN_SENTS} granic, {TRAIN_DOCS} aktow)",
        f"Dev:   spacy_config/dev.spacy ({DEV_SENTS} granic, {DEV_DOCS} aktow)",
        "",
        "| Eksperyment | Width | Window | SENTS_F | SENTS_P | SENTS_R | Model [MB] | Czas [s] | Status |",
        "|-------------|-------|--------|---------|---------|---------|------------|----------|--------|",
    ]
    for r in sorted(results, key=lambda x: x.width):
        f1 = f"{r.sents_f * 100:.2f}%" if r.sents_f is not None else "—"
        p = f"{r.sents_p * 100:.2f}%" if r.sents_p is not None else "—"
        rec = f"{r.sents_r * 100:.2f}%" if r.sents_r is not None else "—"
        size = f"{r.model_size_mb:.1f}" if r.model_size_mb is not None else "—"
        lines.append(
            f"| {r.experiment_id} | {r.width} | {r.window_size} "
            f"| {f1} | {p} | {rec} | {size} | {r.train_time_s:.0f} | {r.status} |"
        )

    ok = [r for r in results if r.sents_f is not None]
    if ok:
        best = max(ok, key=lambda r: r.sents_f or 0)
        lines.extend([
            "",
            f"*Najlepszy F1:* {best.experiment_id} — "
            f"F1={best.sents_f * 100:.2f}%, width={best.width}, "
            f"model={best.model_size_mb} MB",
        ])
    return "\n".join(lines) + "\n"


def _run_experiment(
    width: int,
    *,
    max_steps: int | None,
    skip_existing: bool,
    python_exe: str,
) -> ExperimentResult:
    exp_id = _experiment_id(width)
    out_dir = EXPERIMENTS_DIR / exp_id
    config_path = out_dir / "config.cfg"
    model_best = out_dir / "model-best"

    if skip_existing and (model_best / "meta.json").is_file():
        meta = _read_meta(model_best)
        perf = meta.get("performance", {})
        return ExperimentResult(
            experiment_id=exp_id,
            width=width,
            window_size=WINDOW_SIZE,
            vectors=VECTORS,
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
    _write_config(width, config_path)

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
    print(f"  vectors={VECTORS}, window_size={WINDOW_SIZE}, width={width}", flush=True)
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
                width=width,
                window_size=WINDOW_SIZE,
                vectors=VECTORS,
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
            width=width,
            window_size=WINDOW_SIZE,
            vectors=VECTORS,
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
            width=width,
            window_size=WINDOW_SIZE,
            vectors=VECTORS,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Siatka treningu ws2_sm x width.")
    parser.add_argument(
        "--widths",
        nargs="+",
        type=int,
        default=list(DEFAULT_WIDTHS),
        help="Wartosci width tok2vec (domyslnie: 12 32 64 128)",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not BASE_CONFIG.is_file():
        raise FileNotFoundError(f"Brak config.cfg: {BASE_CONFIG}")
    if not GOLD_TRAIN.is_file() or not GOLD_DEV.is_file():
        raise FileNotFoundError(
            f"Brak danych gold:\n  train: {GOLD_TRAIN}\n  dev: {GOLD_DEV}\n"
            "Wygeneruj je skryptem convert_json_to_spacy.py"
        )

    widths = sorted(set(args.widths))
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Plan eksperymentow (experiments_gold_final):", flush=True)
    print(f"  train: {GOLD_TRAIN}", flush=True)
    print(f"  dev:   {GOLD_DEV}", flush=True)
    print(f"  vectors={VECTORS}, window_size={WINDOW_SIZE}", flush=True)
    for w in widths:
        print(f"  - {_experiment_id(w)}", flush=True)

    if args.dry_run:
        print("\n(dry-run — bez treningu)")
        return

    results: list[ExperimentResult] = []
    python_exe = sys.executable

    for width in widths:
        result = _run_experiment(
            width,
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
