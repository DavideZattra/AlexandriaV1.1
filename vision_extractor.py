import fitz  # PyMuPDF
import base64
import requests
import os
import time

# --- CONFIGURATION ---
PDF_PATH = "./Manuals/UserManual8.2.pdf"
OUTPUT_MD = "./tcpos_master.md"
API_URL = "http://localhost:8080/v1/chat/completions"
DEBUG_FOLDER = "./debug_images" # Qui salveremo le immagini per controllarle

# SYSTEM_PROMPT = """
# You are an expert data entry clerk. Transcribe this manual page into clean, structural Markdown. 
# Use # for main titles and ## for subtitles. 
# Convert any visual tables into Markdown tables. 
# If there is a screenshot, write a brief description of it in [brackets]. 
# Ignore page numbers, headers, and footers. 
# Output ONLY the Markdown text, nothing else. Do not hallucinate.
# """

SYSTEM_PROMPT = """
You are a highly precise OCR and data extraction system. Your task is to transcribe this manual page into clean Markdown.

STRICT RULES:
1. EXACT EXTRACTION: Extract only the exact text you see. Do NOT invent, guess, or hallucinate data under any circumstances, especially inside tables! If you cannot read a cell, leave it empty.
2. TABLES: Reproduce the exact columns and rows you see. 
3. IMAGES: For any map, diagram, or screenshot, write a literal description inside brackets, for example: . Do NOT generate fake image URLs.
4. FORMATTING: Use # for main titles and ## for subtitles. Ignore page numbers.
5. Output ONLY the Markdown text, nothing else.
"""

def image_to_base64(pixmap):
    img_data = pixmap.tobytes("jpeg")
    return base64.b64encode(img_data).decode('utf-8')

def extract_page_with_vision(base64_image, page_num):
    print(f"🤖 Pixtral sta leggendo la Pagina {page_num}...")
    
    payload = {
        "model": "pixtral",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "temperature": 0.0,
        "seed": 42 # Forza l'IA a essere deterministica e meno creativa
    }

    try:
        response = requests.post(API_URL, headers={"Content-Type": "application/json"}, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"❌ Errore API sulla pagina {page_num}: {e}")
        return ""

# def main():
#     if not os.path.exists(DEBUG_FOLDER):
#         os.makedirs(DEBUG_FOLDER)

#     print("--- 🚀 STARTING VISION PIPELINE (DEBUG MODE) ---")
#     doc = fitz.open(PDF_PATH)
#     master_markdown = ""

#     # Estraiamo solo le prime 3 pagine per fare il test, non tutto il manuale!
#     for page_num in range(min(3, len(doc))): 
#         page = doc.load_page(page_num)
        
#         # FIX RISOLUZIONE: Aumentiamo lo zoom da 2.0 a 3.0 (equivalente a circa 300 DPI)
#         zoom_matrix = fitz.Matrix(3.0, 3.0) 
#         pix = page.get_pixmap(matrix=zoom_matrix)
        
#         # SALVATAGGIO DEBUG: Salviamo l'immagine per guardarla con i nostri occhi
#         debug_img_path = f"{DEBUG_FOLDER}/page_{page_num + 1}.jpg"
#         pix.save(debug_img_path)
#         print(f"📸 Immagine salvata in: {debug_img_path}")
        
#         b64_image = image_to_base64(pix)
#         md_text = extract_page_with_vision(b64_image, page_num + 1)
        
#         master_markdown += f"\n\n<!-- PAGE {page_num + 1} -->\n\n" + md_text

#         with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
#             f.write(master_markdown)
            
#         # Diamo 2 secondi di respiro al server per pulire la RAM
#         time.sleep(2)

#     print(f"✨ TEST FINITO! Controlla {OUTPUT_MD}")

#Test pagina 11
def main():
    if not os.path.exists(DEBUG_FOLDER):
        os.makedirs(DEBUG_FOLDER)

    print("--- 🎯 STARTING TARGETED TEST (PAGINA 11) ---")
    doc = fitz.open(PDF_PATH)
    
    # Pagina 11 nel lettore PDF corrisponde all'indice 10 in Python
    target_page_index = 49 
    
    if target_page_index >= len(doc):
        print("⚠️ Il PDF ha meno di 11 pagine!")
        return

    page = doc.load_page(target_page_index)
    
    # Manteniamo la risoluzione altissima (300+ DPI) vitale per le tabelle
    zoom_matrix = fitz.Matrix(4.0, 4.0) 
    pix = page.get_pixmap(matrix=zoom_matrix)
    
    # Salviamo l'immagine di debug
    debug_img_path = f"{DEBUG_FOLDER}/test_pagina_11.jpg"
    pix.save(debug_img_path)
    print(f"📸 Immagine ad alta risoluzione salvata in: {debug_img_path}")
    
    # Estrazione
    b64_image = image_to_base64(pix)
    md_text = extract_page_with_vision(b64_image, 11)
    
    # Salviamo l'output isolato in un file pulito
    output_test_file = "./test_pagina_11.md"
    with open(output_test_file, 'w', encoding='utf-8') as f:
        f.write(f"<!-- TEST PAGINA 11 -->\n\n{md_text}")

    print(f"✨ TEST COMPLETATO! Controlla il file {output_test_file}")

if __name__ == "__main__":
    main()