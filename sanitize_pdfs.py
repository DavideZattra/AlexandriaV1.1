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
        print(f"❌ Errore: File non trovato in {input_path}")
        return

    print(f"🧹 Avvio modulo Data Pre-processing su: {input_path}")
    doc_in = fitz.open(input_path)
    doc_out = fitz.open()  # Crea un nuovo PDF vuoto

    for start, end in page_ranges:
        # Controllo di sicurezza per non sforare i limiti del PDF
        max_page = len(doc_in) - 1
        safe_start = max(0, start)
        safe_end = min(max_page, end)
        
        print(f"➕ Estrazione blocco: pagine da {safe_start + 1} a {safe_end + 1}...")
        
        # insert_pdf copia chirurgicamente le pagine desiderate nel nuovo file
        doc_out.insert_pdf(doc_in, from_page=safe_start, to_page=safe_end)

    doc_out.save(output_path)
    doc_in.close()
    doc_out.close()
    print(f"✨ Successo! PDF pulito e pronto per l'ingestion salvato in: {output_path}")

if __name__ == "__main__":
    print("--- 🏛️ PROGETTO ALEXANDRIA: DATA SANITIZER ---")
    sanitize_pdf(INPUT_PDF, OUTPUT_PDF, PAGES_TO_KEEP)