import os
from langchain_community.document_loaders import DirectoryLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION ---
# 1. Extract your CHM file using 7-Zip into a folder named 'tcpos_manual_html'
DATA_DIRECTORY = "./tcpos_manual_html"
CHROMA_DB_DIR = "./chroma_db"

def main():
    print("--- STARTING TCPOS MANUAL INGESTION ---")

    # 1. LOAD THE DOCUMENTS
    # We use BSHTMLLoader which beautifully strips HTML tags and keeps the text clean
    print(f"Scanning directory: {DATA_DIRECTORY} for HTML files...")
    loader = DirectoryLoader(
        DATA_DIRECTORY,
        glob="**/*.htm*", # Catches .htm and .html
        loader_cls=BSHTMLLoader,
        show_progress=True
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} HTML pages.")

    # 2. IMAGE HANDLING (Architecture Placeholder)
    # ---------------------------------------------------------
    # In Phase 3, we will add a function here that loops through the HTML,
    # finds <img src="X">, sends image X to a local Vision LLM (like LLaVA),
    # gets a description, and appends it to the document's page_content.
    # ---------------------------------------------------------

    # 3. CHUNK THE TEXT
    # We split the text into manageable pieces for your 8B model's context window
    print("Splitting documents into semantic chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Number of characters per chunk
        chunk_overlap=200, # Overlap to maintain context across chunks
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # 4. INITIALIZE EMBEDDINGS
    print("Loading local embedding model (multilingual-e5-base)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={'device': 'cuda'}
    )

    # 5. SAVE TO CHROMADB
    print("Vectorizing and saving to ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    print(f"INGESTION COMPLETE! Database saved in {CHROMA_DB_DIR}")

if __name__ == "__main__":
    main()