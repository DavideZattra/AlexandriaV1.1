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

    print("--- ALEXANDRIA: CHUNKING MODULE ---")

    page_folders = sorted([f for f in os.listdir(SOURCE_DIR) if f.startswith("page_")])
    
    for folder in page_folders:
        folder_path = os.path.join(SOURCE_DIR, folder)
        page_num = int(folder.split("_")[1])
        
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
        
        # Fallback: if the page has no Markdown headers, keep it as a single block
        if not header_splits:
            from langchain_core.documents import Document
            header_splits = [Document(page_content=markdown_content)]
        
        # 2. Secondo split per rispettare i limiti di grandezza
        for doc in header_splits:
            sub_chunks = text_splitter.split_documents([doc])
            
            for chunk in sub_chunks:
                # Inject mandatory RAG metadata
                chunk.metadata["source_page"] = page_num
                chunk.metadata["document_name"] = "tcpos_manual"
                
                all_chunks.append(chunk)
                
    print(f"Processing complete. Generated {len(all_chunks)} chunks ready for the vector database.")
    return all_chunks

if __name__ == "__main__":
    chunks = load_and_chunk_data()
    if chunks:
        print(f"\nSample chunk 1:\nMetadata: {chunks[0].metadata}\nContent:\n{chunks[0].page_content[:300]}...")