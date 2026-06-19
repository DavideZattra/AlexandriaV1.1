"""Embedding-profile helpers for multi-model coexistence.

Lets several embedding models live side by side, each in its own ChromaDB
directory, so their retrieval accuracy can be compared. See EMBEDDING_PROFILES
in config.py for the registry.
"""

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import EMBEDDING_PROFILES, DEFAULT_PROFILE


class PrefixedEmbeddings(Embeddings):
    """Wraps an embedder to prepend an instruction to QUERIES only.

    Models like BGE are trained with an asymmetric prefix on the query side
    (not on documents). Mirroring that here keeps any cross-model comparison
    fair. When query_prefix is empty this is a transparent pass-through.
    """

    def __init__(self, base: Embeddings, query_prefix: str = ""):
        self._base = base
        self._query_prefix = query_prefix

    def embed_documents(self, texts):
        return self._base.embed_documents(texts)

    def embed_query(self, text):
        return self._base.embed_query(self._query_prefix + text)


def get_profile(profile: str = DEFAULT_PROFILE) -> dict:
    if profile not in EMBEDDING_PROFILES:
        raise ValueError(
            f"Unknown embedding profile '{profile}'. "
            f"Available: {list(EMBEDDING_PROFILES)}"
        )
    return EMBEDDING_PROFILES[profile]


def make_embeddings(profile: str = DEFAULT_PROFILE) -> Embeddings:
    """Build the embedding function for a profile, with query prefix applied."""
    cfg = get_profile(profile)
    base = HuggingFaceEmbeddings(model_name=cfg["model"])
    return PrefixedEmbeddings(base, query_prefix=cfg.get("query_prefix", ""))


def load_vector_db(profile: str = DEFAULT_PROFILE) -> Chroma:
    """Open the persisted ChromaDB for a profile with the right embedder."""
    cfg = get_profile(profile)
    return Chroma(
        persist_directory=cfg["path"],
        embedding_function=make_embeddings(profile),
    )
