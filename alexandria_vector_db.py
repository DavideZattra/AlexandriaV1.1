
import os
import shutil
import argparse
from langchain_chroma import Chroma
from alexandria_chunking import load_and_chunk_data
from config import EMBEDDING_PROFILES, DEFAULT_PROFILE
from embeddings import make_embeddings, get_profile

def build_vector_db(profile: str = DEFAULT_PROFILE, chunks=None):
    cfg = get_profile(profile)
    chroma_path = cfg["path"]
    model_name = cfg["model"]

    print(f"--- ALEXANDRIA: VECTOR DATABASE BUILDER (profile: {profile}) ---")

    # 1. Load and chunk the Markdown files (reused across profiles if provided)
    if chunks is None:
        print("\n1. Starting extraction and chunking process...")
        chunks = load_and_chunk_data()

    if not chunks:
        print("ERROR: No chunks generated. Verify that Markdown files exist in the source directory.")
        return

    # 2. Wipe previous database for THIS profile to prevent duplicate entries
    if os.path.exists(chroma_path):
        print(f"\n2. Existing database detected at {chroma_path}. Clearing...")
        shutil.rmtree(chroma_path)

    # 3. Initialize the profile's embedding model
    print(f"\n3. Loading embedding model: {model_name}")
    print("   (First run will download model weights)")
    embeddings = make_embeddings(profile)

    # 4. Vectorize and persist
    print("\n4. Computing vectors and saving to ChromaDB (this may take a few minutes)...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_path,
    )

    print(f"\nSuccess. Knowledge base indexed and saved to: {chroma_path}")

def main():
    parser = argparse.ArgumentParser(description="Build the Alexandria vector database for one or all embedding profiles.")
    parser.add_argument(
        "--profile",
        type=str,
        default=DEFAULT_PROFILE,
        help=f"Embedding profile to build: {', '.join(EMBEDDING_PROFILES)}, or 'all'.",
    )
    args = parser.parse_args()

    if args.profile.lower() == "all":
        # Chunk once, reuse across every profile build
        print("Chunking once for all profiles...")
        chunks = load_and_chunk_data()
        for profile in EMBEDDING_PROFILES:
            build_vector_db(profile, chunks=chunks)
    else:
        build_vector_db(args.profile)

if __name__ == "__main__":
    main()
