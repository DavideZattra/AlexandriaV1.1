import fitz  # PyMuPDF
import base64
import requests
import os
import time

# --- CONFIGURATION ---
PDF_PATH = "./Manuals/UserManual8.2.pdf"
OUTPUT_MD = "./tcpos_master.md"

# Qwen2.5-VL served via llama.cpp on a dedicated port (separate from the
# main text LLM on port 8080, so both can be loaded concurrently if VRAM allows).
API_URL = "http://localhost:8081/v1/chat/completions"
VISION_MODEL = "qwen2.5-vl"
DEBUG_FOLDER = "./debug_images"

SYSTEM_PROMPT = """
You are a highly precise OCR and data extraction system. Your task is to transcribe this manual page into clean Markdown.

STRICT RULES:
1. EXACT EXTRACTION: Extract only the exact text you see. Do NOT invent, guess, or hallucinate data under any circumstances, especially inside tables! If you cannot read a cell, leave it empty.
2. TABLES: Reproduce the exact columns and rows you see.
3. IMAGES: For any map, diagram, or screenshot, write a literal description inside brackets. Do NOT generate fake image URLs.
4. FORMATTING: Use # for main titles and ## for subtitles. Ignore page numbers.
5. Output ONLY the Markdown text, nothing else.
"""

def image_to_base64(pixmap):
    img_data = pixmap.tobytes("jpeg")
    return base64.b64encode(img_data).decode('utf-8')

def extract_page_with_vision(base64_image, page_num):
    print(f"Vision model processing page {page_num}...")

    payload = {
        "model": VISION_MODEL,
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
        "seed": 42  # Forces deterministic output
    }

    try:
        response = requests.post(API_URL, headers={"Content-Type": "application/json"}, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"ERROR: API call failed on page {page_num}: {e}")
        return ""

def main():
    if not os.path.exists(DEBUG_FOLDER):
        os.makedirs(DEBUG_FOLDER)

    print("--- STARTING TARGETED SINGLE-PAGE TEST ---")
    doc = fitz.open(PDF_PATH)

    target_page_index = 49  # 0-indexed; corresponds to page 50 in the PDF reader

    if target_page_index >= len(doc):
        print(f"ERROR: PDF has fewer than {target_page_index + 1} pages.")
        return

    page = doc.load_page(target_page_index)

    # High resolution (~400 DPI) — critical for table accuracy
    zoom_matrix = fitz.Matrix(4.0, 4.0)
    pix = page.get_pixmap(matrix=zoom_matrix)

    debug_img_path = f"{DEBUG_FOLDER}/test_page_{target_page_index + 1}.jpg"
    pix.save(debug_img_path)
    print(f"High-resolution debug image saved: {debug_img_path}")

    b64_image = image_to_base64(pix)
    md_text = extract_page_with_vision(b64_image, target_page_index + 1)

    output_test_file = f"./test_page_{target_page_index + 1}.md"
    with open(output_test_file, 'w', encoding='utf-8') as f:
        f.write(f"<!-- TEST PAGE {target_page_index + 1} -->\n\n{md_text}")

    print(f"Test complete. Output saved to: {output_test_file}")

if __name__ == "__main__":
    main()
