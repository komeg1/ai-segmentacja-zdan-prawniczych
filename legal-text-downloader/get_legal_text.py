import requests
import pdfplumber
import os
import time
import sys
import re

DATA_DIR = "data/acts"
os.makedirs(DATA_DIR, exist_ok=True)

# Use years from command line if provided
if len(sys.argv) > 1:
    YEARS = [int(y) for y in sys.argv[1:]]
else:
    YEARS = range(2015, 2026)

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
    text = re.sub(r"(?<=[^\s])\d+\)", "", text)

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
    median_size = all_sizes[len(all_sizes) // 2]

    for w in words:
        # Check if font is small, matches 'digit)' and is at the bottom
        if (w["size"] < median_size * 0.85 and 
            re.match(r"\d+\)", w["text"]) and 
            w["top"] > page.height * 0.80):
            return w["top"]

    return None

def process_pdf(pdf_path, txt_clean_path, txt_raw_path):
    """Convert PDF to both RAW and CLEAN text files"""
    try:
        clean_pages = []
        raw_pages = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Get raw text
                raw_text = page.extract_text()
                if raw_text:
                    raw_pages.append(raw_text)

                # Get clean text by cutting off footnotes
                cut_y = detect_footnote_cut(page)
                
                if cut_y:
                    # Cut page above the line/footnote
                    target_area = page.crop((0, 0, page.width, cut_y - 2))
                else:
                    # Just remove a small bottom margin
                    target_area = page.crop((0, 0, page.width, page.height * 0.96))

                cleaned = clean_page_text(target_area.extract_text())
                if cleaned:
                    clean_pages.append(cleaned)

        # Save RAW version
        with open(txt_raw_path, "w", encoding="utf-8") as f:
            f.write("\n".join(raw_pages))

        # Save CLEAN version
        final_clean = "\n".join(clean_pages)
        final_clean = re.sub(r"\n\s*\n+", "\n\n", final_clean).strip()
        with open(txt_clean_path, "w", encoding="utf-8") as f:
            f.write(final_clean)

        return True

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return False

def main():
    for year in YEARS:
        print(f"\n--- Processing year {year} ---")
        acts = get_acts(year)
        
        year_dir = os.path.join(DATA_DIR, str(year))
        os.makedirs(year_dir, exist_ok=True)

        for act in acts:
            if not act.get("textPDF"):
                continue

            pos = act.get("pos")
            # Create a safe filename
            title = re.sub(r'[\\/*?:"<>|]', "", act.get("title", "no_title"))[:40].strip()
            
            base_name = f"act_{year}_{pos}_{title}"
            pdf_path = os.path.join(year_dir, f"{base_name}.pdf")
            txt_clean = os.path.join(year_dir, f"{base_name}_clean.txt")
            txt_raw = os.path.join(year_dir, f"{base_name}_raw.txt")

            if os.path.exists(txt_clean):
                continue

            pdf_url = f"https://api.sejm.gov.pl/eli/acts/DU/{year}/{pos}/text.pdf"

            if download_pdf(pdf_url, pdf_path):
                time.sleep(0.1)
                if process_pdf(pdf_path, txt_clean, txt_raw):
                    print(f"  Done: {base_name}")
                
                # os.remove(pdf_path) # Uncomment to delete PDF after conversion

if __name__ == "__main__":
    main()