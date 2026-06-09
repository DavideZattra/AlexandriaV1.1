import fitz  # PyMuPDF
import os

# --- CONFIGURAZIONE ---
INPUT_PDF = "./Manuals/tcpos_manual.pdf"
OUTPUT_PDF = "./Manuals/Sanitized/tcpos_manual.pdf"

# Definisci i range di pagine da TENERE (il "succo" del manuale).
# ⚠️ ATTENZIONE: Python conta partendo da 0! 
# Se nel lettore PDF vedi pagina 12, qui devi scrivere 11.
PAGES_TO_KEEP = [
    (44, 2728),   # Esempio: Tieni le pagine da 12 a 81 del PDF (Capitoli centrali)
    # (85, 120)   # Esempio: Salta 4 pagine inutili e tieni dalla 86 alla 121
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