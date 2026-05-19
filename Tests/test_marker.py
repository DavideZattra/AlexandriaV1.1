import os
import fitz  # PyMuPDF
import subprocess

# --- CONFIGURAZIONE ALEXANDRIA ---
PDF_PATH = "./Manuals/tcpos_manual.pdf"
OUTPUT_DIR = "./alexandria_knowledge_base"
TEMP_PDF = "temp_target_page.pdf"

# Pagina 11 nel lettore PDF corrisponde all'indice 10 in Python
TARGET_PAGE_INDEX = 49 

def extract_single_page(input_pdf, page_index, output_pdf):
    """Estrae una singola pagina per test isolati e deterministici."""
    print(f"✂️ Isolamento della pagina {page_index + 1} in corso...")
    doc = fitz.open(input_pdf)
    
    if page_index >= len(doc):
        raise ValueError("L'indice della pagina supera la lunghezza del PDF.")
        
    doc_out = fitz.open()
    doc_out.insert_pdf(doc, from_page=page_index, to_page=page_index)
    doc_out.save(output_pdf)
    
    doc.close()
    doc_out.close()
    print(f"📄 PDF temporaneo creato: {output_pdf}")

def run_marker(input_pdf, output_dir):
    """Invoca il motore OCR Marker tramite riga di comando."""
    print("\n🚀 Avvio del motore OCR Marker (Surya)...")
    print("⏳ La prima esecuzione scaricherà i modelli OCR (~2GB). Attendi...\n")
    
    # Il comando CLI nativo di Marker
    # command = f"marker_single {input_pdf} {output_dir}"
    command = f"marker_single {input_pdf} --output_dir {output_dir}"
    
    try:
        # subprocess.run blocca lo script finché Marker non ha finito, mostrando l'output nel terminale
        subprocess.run(command, shell=True, check=True)
        print(f"\n✨ Estrazione completata con successo!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Errore critico durante l'esecuzione di Marker: {e}")

def main():
    if not os.path.exists(PDF_PATH):
        print(f"⚠️ File sorgente non trovato: {PDF_PATH}")
        return
        
    print("--- 🏛️ PROGETTO ALEXANDRIA: INGESTION PIPELINE ---")
    
    # 1. Creiamo il frammento PDF
    extract_single_page(PDF_PATH, TARGET_PAGE_INDEX, TEMP_PDF)
    
    # 2. Diamo in pasto il frammento a Marker
    run_marker(TEMP_PDF, OUTPUT_DIR)
    
    # 3. Pulizia del file temporaneo (Opzionale, commentato per debug)
    # if os.path.exists(TEMP_PDF):
    #     os.remove(TEMP_PDF)

if __name__ == "__main__":
    main()