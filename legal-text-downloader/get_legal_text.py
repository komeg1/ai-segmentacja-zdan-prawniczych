import requests
import pdfplumber
import os
import time
import sys

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

def pdf_to_text(pdf_path, txt_path):
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Converted PDF to text: {txt_path}")
        return True
    except Exception as e:
        print(f"Error converting PDF {pdf_path}: {e}")
        return False

def main():
    for year in YEARS:
        print(f"\n--- Processing year {year} ---")
        acts = get_acts(year)
        print(f"Found {len(acts)} acts")

        year_dir = os.path.join(DATA_DIR, str(year))
        os.makedirs(year_dir, exist_ok=True)

        for act in acts:
            if not act.get("textPDF", False):
                continue

            pos = act.get("pos")
            title = act.get("title", "no_title").replace("/", "_").replace(" ", "_")
            pdf_file = os.path.join(year_dir, f"{title}_{pos}.pdf")
            txt_file = pdf_file.replace(".pdf", ".txt")
            pdf_url = f"https://api.sejm.gov.pl/eli/acts/DU/{year}/{pos}/text.pdf"

            if os.path.exists(txt_file):
                print(f"Already exists: {txt_file}, skipping")
                continue

            if download_pdf(pdf_url, pdf_file):
                time.sleep(0.3)
                pdf_to_text(pdf_file, txt_file)
                time.sleep(0.3)

if __name__ == "__main__":
    main()