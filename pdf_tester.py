import os
import pymupdf4llm

# --- CONFIGURATION ---
PDF_FILE_PATH = "./tcpos_manual_pdf/TCPOS.net_UserManual_V8.1.2.pdf"  # Replace with your actual PDF file name
OUTPUT_MD_PATH = "./tcpos_from_pdf.md"

def extract_pdf_to_markdown(pdf_path, output_path):
    print(f"📄 Analyzing PDF: {pdf_path}...")
    
    try:
        # This function does heavy lifting: it tries to reconstruct tables,
        # guess headers based on font sizes, and ignore page numbers/footers.
        md_text = pymupdf4llm.to_markdown(pdf_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_text)
            
        print(f"✨ DONE! Successfully converted PDF to Markdown.")
        print(f"💾 Saved as: {output_path}")
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")

def main():
    if not os.path.exists(PDF_FILE_PATH):
        print(f"⚠️ Could not find {PDF_FILE_PATH}. Please put a PDF in the folder.")
        return
        
    extract_pdf_to_markdown(PDF_FILE_PATH, OUTPUT_MD_PATH)

if __name__ == "__main__":
    main()