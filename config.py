# --- SHARED PROJECT CONFIGURATION ---
# Centralized constants used across ingestion (alexandria_vector_db.py) and
# query time (router.py) to guarantee the embedding space stays consistent.

CHROMA_PATH = "./alexandria_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- TWO-STAGE RETRIEVAL CONFIGURATION ---
# Stage 1 (bi-encoder / ChromaDB): cast a wide net of candidate chunks.
# Stage 2 (cross-encoder reranker): re-score those candidates against the query
# and keep only the most relevant. RERANK_CANDIDATES should be >= RERANK_TOP_K.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 20  # Stage 1: how many chunks ChromaDB returns for reranking
RERANK_TOP_K = 5        # Stage 2: how many chunks survive reranking and feed the LLM

# --- CONTIGUOUS PAGE CONTEXT EXPANSION ---
# After reranking, each surviving chunk is expanded with the chunks from its
# neighbouring pages (within +/- CONTEXT_WINDOW pages), then stitched together
# in page order. This gives the LLM the surrounding context a single chunk often
# lacks (e.g. a procedure that continues onto the next page).
ENABLE_CONTEXT_EXPANSION = True
CONTEXT_WINDOW = 1  # Number of pages to include on each side of a retrieved page
