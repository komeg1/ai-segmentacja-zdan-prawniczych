"""
Eksport bledow segmentacji modelu na dev.spacy do przegladu wzrokowego.

Uzycie (z katalogu spacy/src):
  python export_dev_errors.py
  python export_dev_errors.py --model ../spacy_config/experiments_gold/ws2_sm/model-best
  python export_dev_errors.py --corpus ../spacy_config/dev.spacy --context 120
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import spacy
from spacy.tokens import DocBin

SCRIPT_DIR = Path(__file__).resolve().parent
SPACY_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL = SPACY_ROOT / "spacy_config/experiments_gold/ws2_sm/model-best"
DEFAULT_CORPUS = SPACY_ROOT / "spacy_config/dev.spacy"
DEFAULT_OUTPUT_DIR = SPACY_ROOT / "spacy_config/experiments_gold/ws2_sm/error_analysis"


def _sentence_spans(doc) -> list[tuple[int, int, str]]:
    return [(s.start_char, s.end_char, s.text) for s in doc.sents]


def _context(text: str, char_pos: int, radius: int) -> str:
    start = max(0, char_pos - radius)
    end = min(len(text), char_pos + radius)
    snippet = text[start:end]
    rel = char_pos - start
    return snippet, rel


def _find_sentence_at(spans: list[tuple[int, int, str]], char_pos: int) -> str | None:
    for start, end, sent_text in spans:
        if start <= char_pos < end:
            return sent_text
    return None


def _collect_doc_errors(
    gold_doc,
    pred_doc,
    doc_idx: int,
    *,
    context_radius: int,
) -> list[dict]:
    text = gold_doc.text
    gold_starts = {t.idx for t in gold_doc if t.is_sent_start}
    pred_starts = {t.idx for t in pred_doc if t.is_sent_start}

    gold_spans = _sentence_spans(gold_doc)
    pred_spans = _sentence_spans(pred_doc)

    errors = []
    for char_pos in sorted(gold_starts ^ pred_starts):
        is_false_positive = char_pos in pred_starts and char_pos not in gold_starts
        error_type = "false_positive" if is_false_positive else "false_negative"
        snippet, rel = _context(text, char_pos, context_radius)
        marker = "^^^" if is_false_positive else ">>>"

        errors.append({
            "doc_idx": doc_idx,
            "char_pos": char_pos,
            "error_type": error_type,
            "label_pl": "nadmiarowy podzial" if is_false_positive else "pominiety podzial",
            "context": snippet,
            "marker_offset_in_context": rel,
            "gold_sentence": _find_sentence_at(gold_spans, char_pos),
            "pred_sentence": _find_sentence_at(pred_spans, char_pos),
            "gold_prev_sentence": _find_sentence_at(gold_spans, max(0, char_pos - 1)),
            "pred_prev_sentence": _find_sentence_at(pred_spans, max(0, char_pos - 1)),
            "marker": marker,
        })
    return errors


def _write_html(errors: list[dict], path: Path, model_path: Path, corpus_path: Path) -> None:
    fp = [e for e in errors if e["error_type"] == "false_positive"]
    fn = [e for e in errors if e["error_type"] == "false_negative"]

    def render_block(err: dict) -> str:
        ctx = html.escape(err["context"])
        pos = err["marker_offset_in_context"]
        marked = (
            html.escape(err["context"][:pos])
            + f'<mark class="{err["error_type"]}">'
            + html.escape(err["context"][pos : pos + 1] or "|")
            + "</mark>"
            + html.escape(err["context"][pos + 1 :])
        )
        gold = html.escape(err["gold_sentence"] or "—")
        pred = html.escape(err["pred_sentence"] or "—")
        gold_prev = html.escape(err["gold_prev_sentence"] or "—")
        pred_prev = html.escape(err["pred_prev_sentence"] or "—")

        return f"""
        <div class="error {err['error_type']}">
          <h3>#{err['doc_idx']} @ {err['char_pos']} — {html.escape(err['label_pl'])}</h3>
          <p class="ctx">{marked}</p>
          <table>
            <tr><th></th><th>GOLD</th><th>MODEL</th></tr>
            <tr><td>poprzednie zdanie</td><td>{gold_prev}</td><td>{pred_prev}</td></tr>
            <tr><td>zdanie przy bledzie</td><td>{gold}</td><td>{pred}</td></tr>
          </table>
        </div>
        """

    body = "\n".join(render_block(e) for e in errors)
    content = f"""<!DOCTYPE html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <title>Analiza bledow segmentacji — dev</title>
  <style>
    body {{ font-family: Segoe UI, sans-serif; margin: 24px; max-width: 1100px; }}
    h1 {{ font-size: 1.4rem; }}
    .meta {{ color: #555; margin-bottom: 24px; }}
    .error {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px 0; }}
    .false_positive {{ border-left: 5px solid #c0392b; }}
    .false_negative {{ border-left: 5px solid #2980b9; }}
    .ctx {{ background: #f7f7f7; padding: 10px; white-space: pre-wrap; word-break: break-word; }}
    mark.false_positive {{ background: #f8c9c4; }}
    mark.false_negative {{ background: #c4daf8; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.92rem; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; text-align: left; }}
    th {{ background: #f0f0f0; width: 140px; }}
    .legend span {{ margin-right: 16px; }}
    .fp {{ color: #c0392b; }}
    .fn {{ color: #2980b9; }}
  </style>
</head>
<body>
  <h1>Bledy segmentacji na dev</h1>
  <p class="meta">
    Model: <code>{html.escape(str(model_path))}</code><br>
    Korpus: <code>{html.escape(str(corpus_path))}</code><br>
    Bledy: {len(errors)} (FP: {len(fp)}, FN: {len(fn)})
  </p>
  <p class="legend">
    <span class="fp">■ nadmiarowy podzial (model dzieli, gold nie)</span>
    <span class="fn">■ pominiety podzial (gold dzieli, model nie)</span>
  </p>
  {body}
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def _write_txt(errors: list[dict], path: Path) -> None:
    lines = []
    for i, err in enumerate(errors, 1):
        lines.extend([
            f"{'=' * 72}",
            f"[{i}] doc={err['doc_idx']} pos={err['char_pos']} {err['label_pl']}",
            f"kontekst: ...{err['context']}...",
            f"GOLD: {err['gold_sentence']}",
            f"PRED: {err['pred_sentence']}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def export_errors(
    model_path: Path,
    corpus_path: Path,
    output_dir: Path,
    *,
    context_radius: int = 120,
) -> dict:
    if not model_path.is_dir():
        raise FileNotFoundError(f"Brak modelu: {model_path}")
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Brak korpusu: {corpus_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    nlp = spacy.load(model_path)
    doc_bin = DocBin().from_disk(corpus_path)

    all_errors: list[dict] = []
    doc_stats = []

    for doc_idx, gold_doc in enumerate(doc_bin.get_docs(nlp.vocab)):
        pred_doc = nlp(gold_doc.text)
        doc_errors = _collect_doc_errors(
            gold_doc, pred_doc, doc_idx, context_radius=context_radius
        )
        all_errors.extend(doc_errors)
        doc_stats.append({
            "doc_idx": doc_idx,
            "text_preview": gold_doc.text[:120].replace("\r", " ").replace("\n", " "),
            "errors": len(doc_errors),
            "false_positive": sum(1 for e in doc_errors if e["error_type"] == "false_positive"),
            "false_negative": sum(1 for e in doc_errors if e["error_type"] == "false_negative"),
        })

    summary = {
        "model": str(model_path),
        "corpus": str(corpus_path),
        "documents": len(doc_stats),
        "total_errors": len(all_errors),
        "false_positive": sum(1 for e in all_errors if e["error_type"] == "false_positive"),
        "false_negative": sum(1 for e in all_errors if e["error_type"] == "false_negative"),
        "per_document": doc_stats,
        "errors": all_errors,
    }

    json_path = output_dir / "dev_errors.json"
    html_path = output_dir / "dev_errors.html"
    txt_path = output_dir / "dev_errors.txt"

    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_html(all_errors, html_path, model_path, corpus_path)
    _write_txt(all_errors, txt_path)

    return {
        "json": json_path,
        "html": html_path,
        "txt": txt_path,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Eksport bledow segmentacji z dev.spacy")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--context", type=int, default=120, help="Znakow kontekstu wokol bledu")
    args = parser.parse_args()

    result = export_errors(
        args.model.resolve(),
        args.corpus.resolve(),
        args.output_dir.resolve(),
        context_radius=args.context,
    )
    s = result["summary"]
    print(f"Bledy: {s['total_errors']} (FP={s['false_positive']}, FN={s['false_negative']})")
    print(f"JSON: {result['json']}")
    print(f"HTML: {result['html']}")
    print(f"TXT:  {result['txt']}")


if __name__ == "__main__":
    main()
