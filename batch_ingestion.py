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
    """Legge l'ultimo indice di pagina elaborato con successo."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return -1 # File vuoto o corrotto, ricomincia da zero
    return -1

def update_checkpoint(page_index):
    """Salva la pagina corrente su disco come completata (State Management)."""
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(page_index))

def extract_single_page(doc, page_index, output_pdf):
    """Isola chirurgicamente una pagina per passarla al motore OCR."""
    doc_out = fitz.open()
    doc_out.insert_pdf(doc, from_page=page_index, to_page=page_index)
    doc_out.save(output_pdf)
    doc_out.close()

def main():
    if not os.path.exists(INPUT_PDF):
        print(f"❌ Errore Critico: File sorgente non trovato in {INPUT_PDF}")
        return

    doc = fitz.open(INPUT_PDF)
    total_pages = len(doc)
    
    last_processed = get_last_checkpoint()
    start_page = last_processed + 1

    if start_page >= total_pages:
        print("✨ Il documento è già stato scansionato al 100%! Cancella il file checkpoint.txt se vuoi ricominciare.")
        doc.close()
        return

    print("--- 🏛️ ALEXANDRIA: MOTORE DI INGESTION BATCH ---")
    print(f"📄 Pagine totali rilevate: {total_pages}")
    print(f"▶️ Ripresa dall'indice: {start_page} (Corrisponde alla Pagina {start_page + 1})")
    print(f"⏱️ Interruzione programmata ogni {PAUSE_INTERVAL_SECONDS / 60} minuti.\n")

    session_start_time = time.time()

    for page_idx in range(start_page, total_pages):
        print(f"⚙️ Elaborazione Pagina {page_idx + 1}/{total_pages}...", end="", flush=True)
        
        # 1. Crea la pagina temporanea
        extract_single_page(doc, page_idx, TEMP_PDF)
        
        # 2. Crea una sottocartella specifica (es: /page_0045/) per tenere tutto in ordine
        page_output_dir = os.path.join(OUTPUT_BASE_DIR, f"page_{page_idx + 1:04d}")
        os.makedirs(page_output_dir, exist_ok=True)
        
        # 3. Invoca Marker (silenziamo l'output per non intasare il terminale)
        command = f"marker_single {TEMP_PDF} --output_dir {page_output_dir}"
        
        try:
            # stdout=subprocess.DEVNULL nasconde i log tecnici di Surya, tenendo il terminale pulito
            subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print(f" ✅ Completata")
        except subprocess.CalledProcessError:
            print(f"\n❌ Errore durante Marker sulla pagina {page_idx + 1}.")
            print("L'esecuzione si ferma per permetterti di indagare. Il checkpoint precedente è salvo.")
            sys.exit(1)

        # 4. Registra il progresso su disco (Checkpoint)
        update_checkpoint(page_idx)
        
        # 5. Controllo del Timer (Il Guardiano dei 30 Minuti)
        elapsed_time = time.time() - session_start_time
        if elapsed_time >= PAUSE_INTERVAL_SECONDS:
            print(f"\n⏳ ATTENZIONE: Sono passati {(elapsed_time / 60):.1f} minuti di calcolo continuo.")
            user_input = input("Vuoi continuare con la prossima pagina? (y/n): ").strip().lower()
            
            if user_input == 'y':
                print("▶️ Riprendo l'elaborazione. Resetto il timer...\n")
                session_start_time = time.time()  # Azzera il contatore
            else:
                print(f"🛑 Sospensione manuale. Checkpoint salvato alla pagina {page_idx + 1}.")
                print("La tua GPU può riposare. Al prossimo avvio riprenderà esattamente da qui.")
                sys.exit(0)

    # Pulizia finale a fine ciclo
    doc.close()
    if os.path.exists(TEMP_PDF):
        os.remove(TEMP_PDF)
    
    print("\n🎉 INGESTIONE DEL MANUALE COMPLETATA AL 100%!")

if __name__ == "__main__":
    main()