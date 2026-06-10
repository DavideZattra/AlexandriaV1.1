import os
import fitz  # PyMuPDF
import subprocess

# --- ALEXANDRIA CONFIGURATION ---
PDF_PATH = "./Manuals/tcpos_manual.pdf"
OUTPUT_DIR = "./alexandria_knowledge_base"
TEMP_PDF = "temp_target_page.pdf"

# 0-indexed; corresponds to page 50 in the PDF reader
TARGET_PAGE_INDEX = 49

def extract_single_page(input_pdf, page_index, output_pdf):
    """Extracts a single page for isolated, deterministic testing."""
    print(f"Isolating page {page_index + 1}...")
    doc = fitz.open(input_pdf)

    if page_index >= len(doc):
        raise ValueError("Page index exceeds the PDF length.")

    doc_out = fitz.open()
    doc_out.insert_pdf(doc, from_page=page_index, to_page=page_index)
    doc_out.save(output_pdf)

    doc.close()
    doc_out.close()
    print(f"Temporary PDF created: {output_pdf}")

def run_marker(input_pdf, output_dir):
    """Invokes the Marker OCR engine via the command line."""
    print("\nStarting the Marker OCR engine (Surya)...")
    print("First run will download the OCR models (~2GB). Please wait...\n")

    # Marker's native CLI command
    # command = f"marker_single {input_pdf} {output_dir}"
    command = f"marker_single {input_pdf} --output_dir {output_dir}"

    try:
        # subprocess.run blocks the script until Marker finishes, showing output in the terminal
        subprocess.run(command, shell=True, check=True)
        print(f"\nExtraction completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Critical failure during Marker execution: {e}")

def main():
    if not os.path.exists(PDF_PATH):
        print(f"WARNING: Source file not found: {PDF_PATH}")
        return

    print("--- ALEXANDRIA: INGESTION PIPELINE ---")

    # 1. Create the PDF fragment
    extract_single_page(PDF_PATH, TARGET_PAGE_INDEX, TEMP_PDF)

    # 2. Feed the fragment to Marker
    run_marker(TEMP_PDF, OUTPUT_DIR)

    # 3. Temporary file cleanup (optional, commented out for debugging)
    # if os.path.exists(TEMP_PDF):
    #     os.remove(TEMP_PDF)

if __name__ == "__main__":
    main()
