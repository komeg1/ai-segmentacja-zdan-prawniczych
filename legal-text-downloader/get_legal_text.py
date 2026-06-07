import requests
import pdfplumber
import os
import time
import re
import argparse
import logging
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- LOGGING ---
logging.basicConfig(
    filename="download_errors.log",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(message)s",
)

DATA_DIR = "data/acts"
os.makedirs(DATA_DIR, exist_ok=True)


def ts():
    """Current timestamp string for debug prints."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def thread_name():
    """Short thread name for debug prints."""
    return threading.current_thread().name


def get_acts(year):
    url = f"https://api.sejm.gov.pl/eli/acts/DU/{year}"
    try:
        r = requests.get(url, timeout=10, headers={"Accept": "application/json"})
        r.raise_for_status()
        return r.json().get("items", [])
    except Exception as e:
        logging.error(f"Error fetching acts for {year}: {e}")
        print(f"Error fetching acts for {year}: {e}")
        return []


def download_pdf(pdf_url, filename, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(pdf_url, timeout=15)

            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 5))
                print(f"  [{ts()}] [{thread_name()}] Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            r.raise_for_status()
            with open(filename, "wb") as f:
                f.write(r.content)
            return True

        except requests.exceptions.ConnectionError:
            wait = 2**attempt
            print(
                f"  [{ts()}] [{thread_name()}] Connection error, retrying in {wait}s..."
                f" (attempt {attempt+1}/{retries})"
            )
            time.sleep(wait)
        except Exception as e:
            logging.error(f"Error downloading PDF {pdf_url}: {e}")
            print(f"  [{ts()}] [{thread_name()}] Error downloading {pdf_url}: {e}")
            return False

    return False


def extract_text_without_superscripts(page):
    """Extract text, replacing tables with [TABELA] and skipping superscripts."""
    tables = page.find_tables()
    table_bboxes = sorted([t.bbox for t in tables], key=lambda b: b[1])

    words = page.extract_words(extra_attrs=["size", "top", "bottom"])
    if not words:
        return ""

    sizes = sorted([w["size"] for w in words])
    median_size = sizes[len(sizes) // 2]

    bottoms = sorted([w["bottom"] for w in words])
    median_bottom = bottoms[len(bottoms) // 2]

    def in_table(word):
        for i, (x0, top, x1, bottom) in enumerate(table_bboxes):
            if x0 <= word["x0"] <= x1 and top <= word["top"] <= bottom:
                return i
        return -1

    lines = []
    current_line = []
    prev_bottom = None
    inserted_tables = set()

    for w in words:
        table_idx = in_table(w)
        if table_idx >= 0:
            if table_idx not in inserted_tables:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = []
                lines.append("[TABELA]")
                inserted_tables.add(table_idx)
                prev_bottom = table_bboxes[table_idx][3]
            continue

        if w["size"] < median_size * 0.75 and w["bottom"] < median_bottom - (
            median_size * 0.3
        ):
            continue

        if prev_bottom is not None and w["top"] > prev_bottom + median_size * 0.5:
            if current_line and current_line[-1].endswith("-"):
                current_line[-1] = current_line[-1][:-1]
                current_line.append(w["text"])
            else:
                lines.append(" ".join(current_line))
                current_line = [w["text"]]
            prev_bottom = w["bottom"]
            continue

        current_line.append(w["text"])
        prev_bottom = w["bottom"]

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)


def detect_footnote_cut(page):
    """Return y-coordinate where footnotes start, or None."""
    for line in page.lines:
        width = line["x1"] - line["x0"]
        if 40 < width < 250 and line["top"] > page.height * 0.70:
            return line["top"]

    words = page.extract_words(extra_attrs=["size"])
    if not words:
        return None

    sizes = sorted([w["size"] for w in words])
    median_size = sizes[len(sizes) // 2]

    for w in words:
        if (
            w["size"] < median_size * 0.85
            and re.match(r"\d+\)", w["text"])
            and w["top"] > page.height * 0.80
        ):
            return w["top"]

    return None


def clean_page_text(text):
    """Remove Dz.U. headers and separators from a single page."""
    if not text:
        return ""

    text = re.sub(
        r"DZIENNIK USTAW RZECZYPOSPOLITEJ POLSKIEJ\s*"
        r"Warszawa,\s*dnia\s*\d+\s*\w+\s*\d{4}\s*r\.\s*Poz\.\s*\d+\s*",
        "",
        text,
    )
    text = re.sub(r"Dziennik Ustaw\s*[–-]\s*\d+\s*[–-]\s*Poz\.\s*\d+\s*", "", text)
    text = re.sub(r"^Dziennik Ustaw\s*Poz\.\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Poz\.\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def join_and_fix_hyphens(pages):
    """Join pages and merge hyphenated words split across lines or pages."""
    text = "\n".join(pages)
    # hyphen at end of line followed by lowercase continuation
    text = re.sub(r"-\s*\n\s*([a-ząćęłńóśźż])", r"\1", text)
    # hyphen with space in same line (pdfplumber column artefact)
    text = re.sub(r"-\s+([a-ząćęłńóśźż])", r"\1", text)
    return text


def process_pdf(pdf_path, txt_clean_path, txt_raw_path, save_raw=False):
    """Convert PDF to text files."""
    try:
        clean_pages = []
        raw_pages = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                if save_raw:
                    raw_text = page.extract_text()
                    if raw_text:
                        raw_pages.append(raw_text)

                cut_y = detect_footnote_cut(page)
                target_area = page.crop(
                    (0, 0, page.width, cut_y - 2 if cut_y else page.height * 0.95)
                )

                raw_clean = extract_text_without_superscripts(target_area)
                cleaned = clean_page_text(raw_clean)

                if cleaned:
                    clean_pages.append(cleaned)

        if save_raw and raw_pages:
            with open(txt_raw_path, "w", encoding="utf-8") as f:
                f.write("\n".join(raw_pages))

        if clean_pages:
            final_clean = join_and_fix_hyphens(clean_pages)
            final_clean = re.sub(r"\n\s*\n+", "\n\n", final_clean).strip()
            with open(txt_clean_path, "w", encoding="utf-8") as f:
                f.write(final_clean)

        return True

    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {e}")
        print(f"  [{ts()}] [{thread_name()}] Error processing {pdf_path}: {e}")
        return False


def process_act(act, year_dir, year, save_raw, keep_pdf, debug=False):
    """Download and process a single act. Designed to run in a thread."""
    if not act.get("textPDF"):
        return None

    pos = act.get("pos")
    title = re.sub(r'[\\/*?:"<>|]', "", act.get("title", "no_title"))[:40].strip()

    base_name = f"act_{year}_{pos}_{title}"
    pdf_path = os.path.join(year_dir, f"{base_name}.pdf")
    txt_clean = os.path.join(year_dir, f"{base_name}_clean.txt")
    txt_raw = os.path.join(year_dir, f"{base_name}_raw.txt")

    if os.path.exists(txt_clean):
        if debug:
            print(f"  [{ts()}] [{thread_name()}] SKIP (exists): {base_name}")
        return f"  Skipped (exists): {base_name}"

    pdf_url = f"https://api.sejm.gov.pl/eli/acts/DU/{year}/{pos}/text.pdf"

    if debug:
        print(f"  [{ts()}] [{thread_name()}] START: {base_name}")

    if not download_pdf(pdf_url, pdf_path):
        return f"  FAILED download: {base_name}"

    success = process_pdf(pdf_path, txt_clean, txt_raw, save_raw=save_raw)

    if not keep_pdf:
        try:
            os.remove(pdf_path)
        except Exception as e:
            logging.error(f"Could not remove PDF {pdf_path}: {e}")

    if debug:
        status = "DONE" if success else "FAILED processing"
        print(f"  [{ts()}] [{thread_name()}] {status}: {base_name}")

    if success:
        return f"  Done: {base_name}"
    else:
        return f"  FAILED processing: {base_name}"


def run_thread_test(workers=5):
    """Quick sanity check that threads actually run in parallel."""
    print(f"\n--- Thread test (10 tasks x 0.5s sleep, {workers} workers) ---")
    print(f"    Sequential time would be ~5.0s, parallel should be ~1.0s\n")

    def fake_task(i):
        print(f"  [{ts()}] [{thread_name()}] START task {i}")
        time.sleep(0.5)
        print(f"  [{ts()}] [{thread_name()}] END   task {i}")
        return i

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(fake_task, range(10)))
    elapsed = time.time() - t_start

    print(f"\n  Finished {len(results)} tasks in {elapsed:.2f}s")
    print(f"  Speedup: {(len(results) * 0.5) / elapsed:.1f}x vs sequential")
    print("--- Thread test done ---\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download and process legal acts from the Polish Sejm API."
    )
    parser.add_argument(
        "years",
        type=int,
        nargs="*",
        help="List of years to process (e.g., 2023 2024). Default range: 2015-2025.",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Keep the downloaded source PDF files on disk.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Save the unformatted raw text as a separate '_raw.txt' file.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel download threads (default: 5).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print thread names and timestamps for each task.",
    )
    parser.add_argument(
        "--test-threads",
        action="store_true",
        help="Run a quick thread parallelism test and exit.",
    )

    args = parser.parse_args()

    if args.test_threads:
        run_thread_test(workers=args.workers)
        return

    years_to_process = args.years if args.years else range(2015, 2026)

    for year in years_to_process:
        print(f"\n--- Processing year {year} ---")
        acts = get_acts(year)

        if not acts:
            print(f"  No acts found for {year}, skipping.")
            continue

        year_dir = os.path.join(DATA_DIR, str(year))
        os.makedirs(year_dir, exist_ok=True)

        t_start = time.time()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_act, act, year_dir, year, args.raw, args.pdf, args.debug
                ): act
                for act in acts
            }
            for future in as_completed(futures):
                result = future.result()
                if result and not args.debug:
                    print(result)

        elapsed = time.time() - t_start
        print(
            f"  Year {year} done in {elapsed:.1f}s "
            f"({len(acts)} acts, {args.workers} workers)"
        )

    print("\nAll done! Check download_errors.log for any errors.")


if __name__ == "__main__":
    main()
