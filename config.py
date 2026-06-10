# --- SHARED PROJECT CONFIGURATION ---
# Centralized constants used across ingestion (alexandria_vector_db.py) and
# query time (router.py) to guarantee the embedding space stays consistent.

CHROMA_PATH = "./alexandria_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
