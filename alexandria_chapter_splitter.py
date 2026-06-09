import fitz  # PyMuPDF
import os
import re

# --- ⚙️ CONFIGURAZIONE ---
INPUT_PDF = "./Manuals/tcpos_manual.pdf"
OUTPUT_DIR = "./alexandria_macro_chapters"

# 🛠️ IL FIX DELL'OFFSET: 
# Inserisci qui lo scostamento tra il numero di pagina stampato e la pagina fisica del PDF.
# Esempio: Se l'indice dice "Pagina 31" ma il lettore PDF indica "Pagina 40", l'offset è 9.
PAGE_OFFSET = 39  # <-- Modifica questo numero con il tuo offset reale!

# Mappatura esatta dell'indice fornito
# Formato: ("Nome del Capitolo", Pagina_Inizio_Stampata, Pagina_Fine_Stampata)
CHAPTERS = [
    ("General Concepts of the TCPOS.net Application", 31, 107),
    ("Admin module", 108, 188),
    ("Sale items Ribbon Page", 189, 742),
    ("Outlets Ribbon Page", 743, 1096),
    ("Environment Ribbon Page", 1098, 1238),
    ("Bookkeeping Ribbon Page", 1239, 1306),
    ("System Ribbon Page", 1307, 1437),
    ("Tools Ribbon Page", 1438, 1474),
    ("Scheduler Ribbon page", 1475, 1544),
    ("Reports Ribbon Page", 1545, 1812),
    ("Tasks Ribbon Page", 1813, 1890),
    ("Support Ribbon Page", 1891, 1946),
    ("FrontEnd Module", 1947, 1948),
    ("User Interface", 1949, 1965),
    ("Function Keys", 1966, 2438),
    ("Operator Permission", 2439, 2466),
    ("Printouts", 2467, 2476),
    ("Management of till data", 2477, 2487),
    ("Country Specific - BELGIUM", 2488, 2517),
    ("Country Specific - FRANCE", 2518, 2620),
    ("ADMIN CUSTOMIZATIONS", 2621, 2668),
    ("FRONTEND CUSTOMIZATIONS", 2669, 2690)
]

def sanitize_filename(name):
    """Removes invalid filename characters and replaces spaces with underscores."""
    clean_name = re.sub(r'[\\/*?:"<>|]', "", name)
    return clean_name.replace(" ", "_").strip()

def split_pdf_by_chapters():
    if not os.path.exists(INPUT_PDF):
        print(f"ERROR: Source file not found at {INPUT_PDF}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("--- ALEXANDRIA: MACRO-CHAPTER SPLITTER ---")
    print(f"Opening manual: {INPUT_PDF}")
    print(f"Page offset: +{PAGE_OFFSET}\n")
    
    doc_in = fitz.open(INPUT_PDF)
    max_pages = len(doc_in)

    for index, (title, printed_start, printed_end) in enumerate(CHAPTERS, start=1):
        
        # Allineamento matematico: Pagina Stampata + Offset - 1 (perché PyMuPDF parte da 0)
        real_start_index = (printed_start + PAGE_OFFSET) - 1
        real_end_index = (printed_end + PAGE_OFFSET) - 1
        
        safe_start = max(0, real_start_index)
        safe_end = min(max_pages - 1, real_end_index)
        
        if safe_start > safe_end:
            print(f"WARNING: Skipping '{title}' — invalid range ({printed_start}-{printed_end})")
            continue

        safe_title = sanitize_filename(title)
        output_filename = f"{index:02d}_{safe_title}.pdf"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        print(f"[{output_filename}] | Index: {printed_start}-{printed_end} -> Physical: {safe_start+1}-{safe_end+1} ...", end=" ")
        
        doc_out = fitz.open()
        doc_out.insert_pdf(doc_in, from_page=safe_start, to_page=safe_end)
        # doc_out.save(output_path)
        doc_out.save(output_path, garbage=4, deflate=True, clean=True)
        doc_out.close()
        
        print("OK")

    doc_in.close()
    print(f"\nSplit complete. Chapter files saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    split_pdf_by_chapters()