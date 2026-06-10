import fitz  # PyMuPDF
import os

# --- CONFIGURATION ---
INPUT_PDF = "./Manuals/tcpos_manual.pdf"
OUTPUT_PDF = "./Manuals/Sanitized/tcpos_manual.pdf"

# Define the page ranges to KEEP (the core content of the manual).
# WARNING: Python counts from 0! If you see page 12 in the PDF reader, write 11 here.
PAGES_TO_KEEP = [
    (44, 2728),   # Example: keep the central chapters
    # (85, 120)   # Example: skip a few useless pages and keep pages 86 to 121
]

def sanitize_pdf(input_path, output_path, page_ranges):
    if not os.path.exists(input_path):
        print(f"ERROR: Source file not found at {input_path}")
        return

    print(f"Starting Data Pre-processing on: {input_path}")
    doc_in = fitz.open(input_path)
    doc_out = fitz.open()

    for start, end in page_ranges:
        max_page = len(doc_in) - 1
        safe_start = max(0, start)
        safe_end = min(max_page, end)

        print(f"Extracting block: pages {safe_start + 1} to {safe_end + 1}...")

        doc_out.insert_pdf(doc_in, from_page=safe_start, to_page=safe_end)

    doc_out.save(output_path)
    doc_in.close()
    doc_out.close()
    print(f"Success. Sanitized PDF saved to: {output_path}")

if __name__ == "__main__":
    print("--- ALEXANDRIA: DATA SANITIZER ---")
    sanitize_pdf(INPUT_PDF, OUTPUT_PDF, PAGES_TO_KEEP)