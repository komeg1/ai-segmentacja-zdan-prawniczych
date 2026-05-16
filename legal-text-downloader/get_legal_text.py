import requests
import pdfplumber
import os
import time
import re
import argparse

DATA_DIR = "data/acts"
os.makedirs(DATA_DIR, exist_ok=True)


def get_acts(year):
    url = f"https://api.sejm.gov.pl/eli/acts/DU/{year}"
    try:
        r = requests.get(url, timeout=10, headers={"Accept": "application/json"})
        r.raise_for_status()
        return r.json().get("items", [])
    except Exception as e:
        print(f"Error fetching acts for {year}: {e}")
        return []


def download_pdf(pdf_url, filename):
    try:
        r = requests.get(pdf_url, timeout=15)
        r.raise_for_status()
        with open(filename, "wb") as f:
            f.write(r.content)
        print(f"Downloaded PDF: {filename}")
        return True
    except Exception as e:
        print(f"Error downloading PDF {pdf_url}: {e}")
        return False


def clean_page_text(text):
    if not text:
        return ""

    # 1. Join broken words (e.g. "roz- \nporządzenie" -> "rozporządzenie")
    text = re.sub(r"-\s*\n\s*", "", text)

    # 2. Remove journal headers ("Dziennik Ustaw" and "Poz.")
    text = re.sub(r"^Dziennik Ustaw.*?\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Poz\.\s*\d+.*?\n", "", text, flags=re.MULTILINE)

    # 3. Remove page numbers (e.g. – 2 –)
    text = re.sub(r"^[–-]\s*\d+\s*[–-]\s*$", "", text, flags=re.MULTILINE)

    # 4. Remove footnotes stuck to words (like 'r.2)' or 'act1)')
    # but keep legal points like ' 1)' or ' 2)'
    text = re.sub(r"(?<=[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ])\d+\)", "", text)

    return text


def detect_footnote_cut(page):
    """Find where the footnotes start on the page"""

    # 1. Try to find a horizontal line separator
    for line in page.lines:
        width = line["x1"] - line["x0"]
        y = line["top"]
        if 40 < width < 250 and y > page.height * 0.70:
            return y

    # 2. If no line, look for smaller font size (footnotes are smaller than main text)
    words = page.extract_words(extra_attrs=["size"])
    if not words:
        return None

    # Get the average font size for main text
    all_sizes = sorted([w["size"] for w in words])
    if not all_sizes:
        return None
    median_size = all_sizes[len(all_sizes) // 2]

    for w in words:
        # Check if font is small, matches 'digit)' and is at the bottom
        if (
            w["size"] < median_size * 0.85
            and re.match(r"\d+\)", w["text"])
            and w["top"] > page.height * 0.80
        ):
            return w["top"]

    return None


def process_pdf(pdf_path, txt_clean_path, txt_raw_path, save_raw=False):
    """Convert PDF to text files based on flags"""
    try:
        clean_pages = []
        raw_pages = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Get raw text if requested
                if save_raw:
                    raw_text = page.extract_text()
                    if raw_text:
                        raw_pages.append(raw_text)

                # Get clean text by cutting off footnotes
                cut_y = detect_footnote_cut(page)

                if cut_y:
                    target_area = page.crop((0, 0, page.width, cut_y - 2))
                else:
                    target_area = page.crop((0, 0, page.width, page.height * 0.96))

                cleaned = clean_page_text(target_area.extract_text())
                if cleaned:
                    clean_pages.append(cleaned)

        # Save RAW version only if flag is present
        if save_raw and raw_pages:
            with open(txt_raw_path, "w", encoding="utf-8") as f:
                f.write("\n".join(raw_pages))

        # Save CLEAN version (always)
        if clean_pages:
            final_clean = "\n".join(clean_pages)
            final_clean = re.sub(r"\n\s*\n+", "\n\n", final_clean).strip()
            with open(txt_clean_path, "w", encoding="utf-8") as f:
                f.write(final_clean)

        return True

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return False


def main():
    # CLI arguments configuration in English
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

    args = parser.parse_args()

    # Fallback to default range if no years are specified
    years_to_process = args.years if args.years else range(2015, 2026)

    for year in years_to_process:
        print(f"\n--- Processing year {year} ---")
        acts = get_acts(year)

        year_dir = os.path.join(DATA_DIR, str(year))
        os.makedirs(year_dir, exist_ok=True)

        for act in acts:
            if not act.get("textPDF"):
                continue

            pos = act.get("pos")
            title = re.sub(r'[\\/*?:"<>|]', "", act.get("title", "no_title"))[
                :40
            ].strip()

            base_name = f"act_{year}_{pos}_{title}"
            pdf_path = os.path.join(year_dir, f"{base_name}.pdf")
            txt_clean = os.path.join(year_dir, f"{base_name}_clean.txt")
            txt_raw = os.path.join(year_dir, f"{base_name}_raw.txt")

            if os.path.exists(txt_clean):
                continue

            pdf_url = f"https://api.sejm.gov.pl/eli/acts/DU/{year}/{pos}/text.pdf"

            if download_pdf(pdf_url, pdf_path):
                time.sleep(0.1)
                if process_pdf(pdf_path, txt_clean, txt_raw, save_raw=args.raw):
                    print(f"  Done: {base_name}")

                # Delete PDF if the user did not pass the --pdf flag
                if not args.pdf:
                    try:
                        os.remove(pdf_path)
                    except Exception as e:
                        print(f"Could not remove PDF {pdf_path}: {e}")


if __name__ == "__main__":
    main()
