import os
import sys
import time
import fitz  # PyMuPDF
import subprocess

# --- ⚙️ CONFIGURAZIONE ARCHITETTURA ---
INPUT_PDF = "./Manuals/Sanitized/tcpos_manual.pdf"
OUTPUT_BASE_DIR = "./alexandria_knowledge_base"
TEMP_PDF = "temp_batch_page.pdf"
CHECKPOINT_FILE = "alexandria_checkpoint.txt"

# Imposta il timer in secondi (30 minuti = 1800 secondi)
PAUSE_INTERVAL_SECONDS = 120 * 60 

def get_last_checkpoint():
    """Reads the last successfully processed page index from the checkpoint file."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return -1  # Empty or corrupted file — restart from the beginning
    return -1

def update_checkpoint(page_index):
    """Persists the current page index to disk as completed (state management)."""
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(page_index))

def extract_single_page(doc, page_index, output_pdf):
    """Extracts a single page from the source document into a temporary PDF for OCR."""
    doc_out = fitz.open()
    doc_out.insert_pdf(doc, from_page=page_index, to_page=page_index)
    doc_out.save(output_pdf)
    doc_out.close()

def main():
    if not os.path.exists(INPUT_PDF):
        print(f"CRITICAL ERROR: Source file not found at {INPUT_PDF}")
        return

    doc = fitz.open(INPUT_PDF)
    total_pages = len(doc)

    last_processed = get_last_checkpoint()
    start_page = last_processed + 1

    if start_page >= total_pages:
        print("Document already 100% processed. Delete checkpoint.txt to restart from scratch.")
        doc.close()
        return

    print("--- ALEXANDRIA: BATCH INGESTION ENGINE ---")
    print(f"Total pages detected: {total_pages}")
    print(f"Resuming from index: {start_page} (Page {start_page + 1})")
    print(f"Scheduled pause every {PAUSE_INTERVAL_SECONDS / 60:.0f} minutes.\n")

    session_start_time = time.time()

    for page_idx in range(start_page, total_pages):
        print(f"Processing page {page_idx + 1}/{total_pages}...", end="", flush=True)

        # 1. Extract the single page into a temporary PDF
        extract_single_page(doc, page_idx, TEMP_PDF)

        # 2. Create a dedicated output subfolder (e.g. /page_0045/)
        page_output_dir = os.path.join(OUTPUT_BASE_DIR, f"page_{page_idx + 1:04d}")
        os.makedirs(page_output_dir, exist_ok=True)

        # 3. Invoke Marker (stdout suppressed to keep terminal clean)
        command = f"marker_single {TEMP_PDF} --output_dir {page_output_dir}"

        try:
            subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print(f" Done")
        except subprocess.CalledProcessError:
            print(f"\nERROR: Marker failed on page {page_idx + 1}.")
            print("Execution halted. Previous checkpoint is preserved — investigation is safe.")
            sys.exit(1)

        # 4. Persist progress to disk
        update_checkpoint(page_idx)

        # 5. Timer check — prompt user to continue after long GPU sessions
        elapsed_time = time.time() - session_start_time
        if elapsed_time >= PAUSE_INTERVAL_SECONDS:
            print(f"\nWARNING: {(elapsed_time / 60):.1f} minutes of continuous processing elapsed.")
            user_input = input("Continue to the next page? (y/n): ").strip().lower()

            if user_input == 'y':
                print("Resuming. Timer reset.\n")
                session_start_time = time.time()
            else:
                print(f"Manual pause. Checkpoint saved at page {page_idx + 1}.")
                print("Restart the script to resume from exactly this point.")
                sys.exit(0)

    # Final cleanup
    doc.close()
    if os.path.exists(TEMP_PDF):
        os.remove(TEMP_PDF)

    print("\nIngestion complete. All pages processed successfully.")

if __name__ == "__main__":
    main()