import os
import glob

def cleanup_pdfs(years):
    base_dir = "data/acts"
    for year in years:
        year_dir = os.path.join(base_dir, str(year))
        if not os.path.exists(year_dir):
            print(f"Directory {year_dir} does not exist, skipping.")
            continue
        
        pdf_files = glob.glob(os.path.join(year_dir, "*.pdf"))
        for pdf_file in pdf_files:
            try:
                os.remove(pdf_file)
                print(f"Deleted: {pdf_file}")
            except Exception as e:
                print(f"Error deleting {pdf_file}: {e}")

if __name__ == "__main__":
    # Clean up PDFs for years 2018-2020
    cleanup_pdfs(range(2018, 2021))