# --- SHARED PROJECT CONFIGURATION ---
# Centralized constants used across ingestion (ingestion/vector_db.py) and
# query time (agent/graph.py) to guarantee the embedding space stays consistent.

from alexandria.paths import CHROMA_DIR, CHROMA_BGE_DIR

# --- EMBEDDING PROFILES ---
# Each profile is a self-contained (model, db_path, query_prefix) bundle so that
# multiple embedding models can coexist and be compared. Models with different
# output dimensions (MiniLM=384, bge-large=1024) CANNOT share a ChromaDB
# collection, so each profile gets its own persisted database directory.
#
# query_prefix: some models (BGE) are trained with an asymmetric instruction
# prepended to queries only. Applying it keeps the accuracy comparison fair.
EMBEDDING_PROFILES = {
    "minilm": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "path": str(CHROMA_DIR),          # existing DB — reused, no rebuild needed
        "query_prefix": "",
    },
    "bge": {
        "model": "BAAI/bge-large-en-v1.5",
        "path": str(CHROMA_BGE_DIR),
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
}
DEFAULT_PROFILE = "bge"

# Backward-compatible aliases derived from the default profile. Existing imports
# (agent/graph.py, ingestion/vector_db.py) keep working unchanged.
CHROMA_PATH = EMBEDDING_PROFILES[DEFAULT_PROFILE]["path"]
EMBEDDING_MODEL = EMBEDDING_PROFILES[DEFAULT_PROFILE]["model"]

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

# --- AGENTIC LOOPS ---
# Self-RAG grounding check: after generation, a verdict node asks whether every
# claim in the answer is supported by the retrieved context. If not, the answer
# is regenerated once with a stricter prompt; if it still fails, it is returned
# with an explicit caveat. The cap prevents infinite loops (each retry is a
# full LLM call).
GROUNDING_MAX_RETRIES = 1

# CRAG document grading: after retrieval, a verdict node asks whether the
# retrieved context is sufficient to answer the question. If not, the query is
# rewritten (expanded acronyms, precise manual terminology) and retrieval runs
# again, up to this many rewrites, before generating from whatever was found.
RETRIEVAL_MAX_RETRIES = 2
