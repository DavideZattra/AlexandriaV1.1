
import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from alexandria.ingestion.chunking import load_and_chunk_data
from alexandria.config import CHROMA_PATH, EMBEDDING_MODEL

def build_vector_db():
    print("--- ALEXANDRIA: VECTOR DATABASE BUILDER ---")

    # 1. Load and chunk the Markdown files
    print("\n1. Starting extraction and chunking process...")
    chunks = load_and_chunk_data()

    if not chunks:
        print("ERROR: No chunks generated. Verify that Markdown files exist in the source directory.")
        return

    # 2. Wipe previous database to prevent duplicate entries on re-runs
    if os.path.exists(CHROMA_PATH):
        print(f"\n2. Existing database detected at {CHROMA_PATH}. Clearing...")
        shutil.rmtree(CHROMA_PATH)

    # 3. Initialize local embedding model
    print(f"\n3. Loading embedding model: {EMBEDDING_MODEL}")
    print("   (First run will download model weights, approx. 100MB)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 4. Vectorize and persist
    print("\n4. Computing vectors and saving to ChromaDB (this may take a few minutes)...")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"\nSuccess. Knowledge base indexed and saved to: {CHROMA_PATH}")

if __name__ == "__main__":
    build_vector_db()