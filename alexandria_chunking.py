import os
import glob
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# --- CONFIGURAZIONE ---
SOURCE_DIR = "./alexandria_knowledge_base"
CHUNK_SIZE = 1000  # Dimensione ideale del frammento in caratteri
CHUNK_OVERLAP = 200  # Sovrapposizione per non perdere il contesto tra i tagli

def load_and_chunk_data():
    all_chunks = []
    
    # Definiamo gli header Markdown su cui fare lo split logico
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    print("--- 🏛️ ALEXANDRIA: MODULO CHUNKING ---")
    
    # Naviga nelle cartelle delle pagine in ordine
    page_folders = sorted([f for f in os.listdir(SOURCE_DIR) if f.startswith("page_")])
    
    for folder in page_folders:
        folder_path = os.path.join(SOURCE_DIR, folder)
        page_num = int(folder.split("_")[1])
        
        # 🔍 IL FIX: Cerca il file .md in tutte le sottocartelle (Effetto Radar)
        md_files = glob.glob(os.path.join(folder_path, "**", "*.md"), recursive=True)
        
        if not md_files:
            continue
            
        md_file_path = md_files[0]
        
        with open(md_file_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()
            
        # Se la pagina è completamente vuota, saltala
        if not markdown_content.strip():
            continue

        # 1. Primo split basato sulla logica dei titoli Markdown
        header_splits = markdown_splitter.split_text(markdown_content)
        
        # Fallback di sicurezza: se la pagina non ha titoli Markdown, tieni il blocco intero
        if not header_splits:
            from langchain_core.documents import Document
            header_splits = [Document(page_content=markdown_content)]
        
        # 2. Secondo split per rispettare i limiti di grandezza
        for doc in header_splits:
            sub_chunks = text_splitter.split_documents([doc])
            
            for chunk in sub_chunks:
                # Iniettiamo i metadati fondamentali per il RAG
                chunk.metadata["source_page"] = page_num
                chunk.metadata["document_name"] = "tcpos_manual"
                
                all_chunks.append(chunk)
                
    print(f"✨ Elaborazione completata! Generati {len(all_chunks)} frammenti di testo pronti per il Vector DB.")
    return all_chunks

if __name__ == "__main__":
    # Test di funzionamento del modulo
    chunks = load_and_chunk_data()
    if chunks:
        print(f"\n📝 Esempio Frammento 1:\nMetadati: {chunks[0].metadata}\nContenuto:\n{chunks[0].page_content[:300]}...")