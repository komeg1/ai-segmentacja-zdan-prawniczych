"""
Experiment grid: pl_core_news_sm / md / lg × window_size on GOLD data.

Gold only (no silver):
  train → spacy_config/train.spacy
  dev   → spacy_config/dev.spacy

Run (from spacy/src):
  python run_gold_experiments.py
  python run_gold_experiments.py --windows 1 3 5 --vectors sm lg
  python run_gold_experiments.py --max-steps 2000   # quick test
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

import spacy
from spacy.tokens import DocBin

from act_text_utils import LineEndingMode, prepare_act_text_from_file
from convert_json_to_spacy import act_year_dirs, convert_train_dev

SCRIPT_DIR = Path(__file__).resolve().parent
SPACY_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SPACY_ROOT.parent
CONFIG_DIR = SPACY_ROOT / "spacy_config"
BASE_CONFIG = CONFIG_DIR / "config.cfg"

DATA_DIR = SPACY_ROOT / "data" / "exported_sentences_ls"
DEFAULT_TRAIN_JSON = DATA_DIR / "exported_sentences_ls_merged.json"
DEFAULT_DEV_JSON = DATA_DIR / "wyciete_zdania_raw_2025.json"
TRAIN_ACT_YEARS = ("2016", "2018", "2019", "2020", "2021")
DEV_ACT_YEARS = ("2025",)
DEFAULT_BENCHMARK_FILE = (
    REPO_ROOT
    / "legal-text-downloader/data/acts/2019"
    / "act_2019_55_Ustawa z dnia 6 grudnia 2018 r. o Krajow_clean.txt"
)

EXPERIMENTS_DIR = CONFIG_DIR / "experiments_gold"
SUMMARY_MD = EXPERIMENTS_DIR / "porownanie_gold_window_vectors.md"
SUMMARY_JSON = EXPERIMENTS_DIR / "porownanie_gold_window_vectors.json"

GOLD_TRAIN = CONFIG_DIR / "train.spacy"
GOLD_DEV = CONFIG_DIR / "dev.spacy"

DEFAULT_VECTORS = ("pl_core_news_sm", "pl_core_news_md", "pl_core_news_lg")
DEFAULT_WINDOWS = (1, 2, 3, 4, 5)


@dataclass
class CorpusStats:
    docs: int
    sents: int


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
    segment_time_s: float | None = None


def _gold_paths(line_ending_mode: LineEndingMode, inicjaly: bool) -> tuple[Path, Path, Path]:
    """Train/dev and experiment directories depending on text mode and tokenizer."""
    if line_ending_mode == "lf" and not inicjaly:
        data_dir = CONFIG_DIR
        exp_dir = CONFIG_DIR / "experiments_gold"
        suffix = ""
    else:
        parts = [line_ending_mode]
        if inicjaly:
            parts.append("inicjaly")
        suffix = "_".join(parts)
        data_dir = CONFIG_DIR / f"gold_{suffix}"
        exp_dir = CONFIG_DIR / f"experiments_gold_{suffix}"
    return data_dir / "train.spacy", data_dir / "dev.spacy", exp_dir


def _ensure_gold_corpus(
    line_ending_mode: LineEndingMode,
    inicjaly: bool,
) -> tuple[Path, Path]:
    train_path, dev_path, _ = _gold_paths(line_ending_mode, inicjaly)
    if train_path.is_file() and dev_path.is_file():
        return train_path, dev_path

    train_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Converting JSON -> train/dev.spacy "
        f"(line_ending={line_ending_mode}, inicjaly={inicjaly})",
        flush=True,
    )
    convert_train_dev(
        DEFAULT_TRAIN_JSON,
        DEFAULT_DEV_JSON,
        train_text_dirs=act_year_dirs(REPO_ROOT, TRAIN_ACT_YEARS),
        dev_text_dirs=act_year_dirs(REPO_ROOT, DEV_ACT_YEARS),
        output_dir=train_path.parent,
        inicjaly=inicjaly,
        line_ending_mode=line_ending_mode,
    )
    return train_path, dev_path


def _corpus_stats(spacy_path: Path) -> CorpusStats:
    nlp = spacy.blank("pl")
    docs = list(DocBin().from_disk(spacy_path).get_docs(nlp.vocab))
    sents = sum(1 for doc in docs for token in doc if token.is_sent_start)
    return CorpusStats(docs=len(docs), sents=sents)


def _short_vectors(name: str) -> str:
    return name.replace("pl_core_news_", "")


def _experiment_id(vectors: str, window_size: int) -> str:
    return f"ws{window_size}_{_short_vectors(vectors)}"


def _pretrained_vectors_for(vectors: str) -> bool:
    """sm has no static vectors — HashEmbedCNN.pretrained_vectors must be false."""
    return _short_vectors(vectors) in ("md", "lg")


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
    use_pretrained = _pretrained_vectors_for(vectors)
    text = re.sub(
        r"^pretrained_vectors = \S+$",
        f"pretrained_vectors = {'true' if use_pretrained else 'false'}",
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


def _previous_train_times() -> dict[str, float]:
    if not SUMMARY_JSON.is_file():
        return {}
    try:
        rows = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {
        row["experiment_id"]: row["train_time_s"]
        for row in rows
        if row.get("train_time_s")
    }


def _resolve_model_dir(model_dir: Path) -> Path | None:
    for name in ("model-best", "model-last"):
        candidate = model_dir / name
        if candidate.is_dir():
            return candidate
    return None


def _measure_segment_time(model_dir: Path, text: str) -> float | None:
    model_path = _resolve_model_dir(model_dir)
    if model_path is None:
        return None
    start = time.perf_counter()
    nlp = spacy.load(model_path)
    _ = nlp(text)
    return time.perf_counter() - start


def _benchmark_segmentation(
    results: list[ExperimentResult],
    benchmark_file: Path,
    *,
    line_ending_mode: LineEndingMode = "lf",
) -> None:
    if not benchmark_file.is_file():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")

    text, _ = prepare_act_text_from_file(benchmark_file, line_ending_mode)
    print(
        f"\nSegmentation benchmark: {benchmark_file.name} "
        f"({len(text):,} characters, {len(results)} models)",
        flush=True,
    )

    for result in results:
        model_dir = Path(result.output_dir)
        if result.status.startswith("skipped") and result.train_time_s == 0.0:
            prev = _previous_train_times()
            if result.experiment_id in prev:
                result.train_time_s = prev[result.experiment_id]

        if _resolve_model_dir(model_dir) is None:
            print(f"  {result.experiment_id}: no model-best/model-last — skipping", flush=True)
            continue

        try:
            result.segment_time_s = _measure_segment_time(model_dir, text)
        except Exception as exc:
            print(f"  {result.experiment_id}: segmentation error — {exc}", flush=True)
            result.segment_time_s = None
            continue

        print(
            f"  {result.experiment_id}: segmentation {result.segment_time_s:.3f}s",
            flush=True,
        )


def _format_table(
    results: list[ExperimentResult],
    train_stats: CorpusStats,
    dev_stats: CorpusStats,
    benchmark_file: Path,
    *,
    gold_train: Path,
    gold_dev: Path,
    line_ending_mode: LineEndingMode,
    inicjaly: bool,
) -> str:
    mode_label = "spaces" if line_ending_mode == "space" else "\\n"
    ini_label = "yes" if inicjaly else "no"
    lines = [
        "# GOLD experiment comparison — vectors × window_size",
        "",
        f"Text: {mode_label}, initials: {ini_label}",
        "",
        f"Train: {gold_train.relative_to(SPACY_ROOT)} "
        f"({train_stats.sents} sentence boundaries, {train_stats.docs} acts)",
        f"Dev:   {gold_dev.relative_to(SPACY_ROOT)} "
        f"({dev_stats.sents} sentence boundaries, {dev_stats.docs} acts)",
        "",
        f"Segmentation benchmark: `{benchmark_file.name}`",
        "",
        "| Experiment | Vectors | Window | SENTS_F | SENTS_P | SENTS_R | Model [MB] | Training [s] | Segmentation [s] | Status |",
        "|------------|---------|--------|---------|---------|---------|------------|--------------|------------------|--------|",
    ]
    for r in sorted(results, key=lambda x: (x.window_size, x.vectors)):
        f1 = f"{r.sents_f * 100:.2f}%" if r.sents_f is not None else "—"
        p = f"{r.sents_p * 100:.2f}%" if r.sents_p is not None else "—"
        rec = f"{r.sents_r * 100:.2f}%" if r.sents_r is not None else "—"
        size = f"{r.model_size_mb:.1f}" if r.model_size_mb is not None else "—"
        train_t = f"{r.train_time_s:.0f}" if r.train_time_s else "—"
        seg_t = f"{r.segment_time_s:.3f}" if r.segment_time_s is not None else "—"
        lines.append(
            f"| {r.experiment_id} | {_short_vectors(r.vectors)} | {r.window_size} "
            f"| {f1} | {p} | {rec} | {size} | {train_t} | {seg_t} | {r.status} |"
        )

    ok = [r for r in results if r.sents_f is not None]
    if ok:
        best = max(ok, key=lambda r: r.sents_f or 0)
        seg_note = (
            f"{best.segment_time_s:.3f}s"
            if best.segment_time_s is not None
            else "—"
        )
        lines.extend([
            "",
            f"*Best F1:* {best.experiment_id} — "
            f"F1={best.sents_f * 100:.2f}%, window={best.window_size}, "
            f"vectors={_short_vectors(best.vectors)}, model={best.model_size_mb} MB, "
            f"training={best.train_time_s:.0f}s, segmentation={seg_note}",
        ])
    return "\n".join(lines) + "\n"


def _run_experiment(
    vectors: str,
    window_size: int,
    *,
    gold_train: Path,
    gold_dev: Path,
    experiments_dir: Path,
    max_steps: int | None,
    skip_existing: bool,
    python_exe: str,
) -> ExperimentResult:
    exp_id = _experiment_id(vectors, window_size)
    out_dir = experiments_dir / exp_id
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
        str(gold_train.resolve()),
        "--paths.dev",
        str(gold_dev.resolve()),
    ]
    if max_steps is not None:
        cmd.extend(["--training.max_steps", str(max_steps)])

    print(f"\n{'=' * 60}", flush=True)
    print(f"Experiment: {exp_id}", flush=True)
    print(
        f"  vectors={vectors}, window_size={window_size}, "
        f"pretrained_vectors={_pretrained_vectors_for(vectors)}",
        flush=True,
    )
    print(f"  output={out_dir}", flush=True)
    print(" ".join(cmd), flush=True)

    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=gold_train.parent, check=False, capture_output=True, text=True)
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
    parser = argparse.ArgumentParser(description="Senter training grid on gold data.")
    parser.add_argument(
        "--vectors",
        nargs="+",
        default=list(DEFAULT_VECTORS),
        help="Vector packages: sm md lg or full pl_core_news_* names",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=list(DEFAULT_WINDOWS),
        help="window_size values to test",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max_steps (default from config.cfg: 20000)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments with existing model-best/meta.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without running training",
    )
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=DEFAULT_BENCHMARK_FILE,
        help="_clean.txt file for segmentation timing after training",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Skip training; load results from JSON and measure segmentation only",
    )
    parser.add_argument(
        "--line-ending-mode",
        choices=("lf", "space"),
        default="lf",
        help="lf: text with \\n; space: \\r\\n -> space (default lf)",
    )
    parser.add_argument(
        "--inicjaly",
        action="store_true",
        help="Add tokenizer exceptions for initials (D., Sz., ...)",
    )
    parser.add_argument(
        "--reconvert",
        action="store_true",
        help="Force re-conversion of JSON -> train/dev.spacy",
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
    line_ending_mode: LineEndingMode = args.line_ending_mode
    inicjaly: bool = args.inicjaly

    if not BASE_CONFIG.is_file():
        raise FileNotFoundError(f"config.cfg not found: {BASE_CONFIG}")

    gold_train, gold_dev, experiments_dir = _gold_paths(line_ending_mode, inicjaly)
    summary_md = experiments_dir / "porownanie_gold_window_vectors.md"
    summary_json = experiments_dir / "porownanie_gold_window_vectors.json"

    if args.reconvert:
        for name in ("train.spacy", "dev.spacy"):
            p = gold_train.parent / name
            if p.is_file():
                p.unlink()

    gold_train, gold_dev = _ensure_gold_corpus(line_ending_mode, inicjaly)

    train_stats = _corpus_stats(gold_train)
    dev_stats = _corpus_stats(gold_dev)

    experiments_dir.mkdir(parents=True, exist_ok=True)
    plan = [(v, w) for v in vectors_list for w in windows]

    mode_label = "spaces" if line_ending_mode == "space" else "\\n"
    print("Experiment plan (GOLD only):", flush=True)
    print(f"  text: {mode_label}, initials: {'yes' if inicjaly else 'no'}", flush=True)
    print("  pretrained_vectors: sm=false, md/lg=true", flush=True)
    print(
        f"  train: {gold_train} "
        f"({train_stats.sents} boundaries, {train_stats.docs} acts)",
        flush=True,
    )
    print(
        f"  dev:   {gold_dev} "
        f"({dev_stats.sents} boundaries, {dev_stats.docs} acts)",
        flush=True,
    )
    print(f"  results: {experiments_dir}", flush=True)
    for v, w in plan:
        print(f"  - {_experiment_id(v, w)}", flush=True)

    benchmark_file = args.benchmark_file.resolve()

    if args.dry_run:
        print("\n(dry-run — no training)")
        return

    results: list[ExperimentResult] = []

    if args.benchmark_only:
        if not summary_json.is_file():
            raise FileNotFoundError(
                f"{summary_json} not found — run full training first or omit --benchmark-only"
            )
        raw = json.loads(summary_json.read_text(encoding="utf-8"))
        results = [ExperimentResult(**row) for row in raw]
        print(f"Loaded {len(results)} results from {summary_json}", flush=True)
    else:
        python_exe = sys.executable
        for vectors, window_size in plan:
            result = _run_experiment(
                vectors,
                window_size,
                gold_train=gold_train,
                gold_dev=gold_dev,
                experiments_dir=experiments_dir,
                max_steps=args.max_steps,
                skip_existing=args.skip_existing,
                python_exe=python_exe,
            )
            results.append(result)
            if result.sents_f is not None:
                print(
                    f"[TRAINING COMPLETE] {result.experiment_id} | "
                    f"F1={result.sents_f * 100:.2f}% "
                    f"P={result.sents_p * 100:.2f}% R={result.sents_r * 100:.2f}% | "
                    f"model={result.model_size_mb} MB | time={result.train_time_s:.0f}s",
                    flush=True,
                )
            else:
                print(f"[TRAINING COMPLETE] {result.experiment_id} | {result.status}", flush=True)
                if result.error:
                    print(f"  error: {result.error[:500]}", flush=True)

    _benchmark_segmentation(
        results, benchmark_file, line_ending_mode=line_ending_mode
    )

    summary_json.write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    table = _format_table(
        results,
        train_stats,
        dev_stats,
        benchmark_file,
        gold_train=gold_train,
        gold_dev=gold_dev,
        line_ending_mode=line_ending_mode,
        inicjaly=inicjaly,
    )
    summary_md.write_text(table, encoding="utf-8")

    print(f"\n{table}")
    print(f"Saved: {summary_md}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()
