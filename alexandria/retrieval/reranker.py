"""Two-stage retrieval: cross-encoder reranker (Stage 2).

The bi-encoder retrieval in ChromaDB (Stage 1) is fast but coarse — it embeds
the query and each document independently, so it can rank loosely-related chunks
highly. A cross-encoder (Stage 2) instead processes the (query, document) pair
jointly in a single forward pass, producing a far more accurate relevance score
at the cost of speed. We therefore retrieve a wide candidate pool with the
bi-encoder, then rerank it down to the best few with the cross-encoder.

The model runs locally (sentence-transformers / CrossEncoder), in keeping with
the project's 100%-local, no-cloud design. It is loaded lazily so importing this
module (or router.py) does not pay the model-load cost until a rerank is needed.
"""

from sentence_transformers import CrossEncoder

from alexandria.config import RERANKER_MODEL


class CrossEncoderReranker:
    """Reranks candidate documents against a query using a local cross-encoder."""

    def __init__(self, model_name: str = RERANKER_MODEL, device: str | None = None):
        self.model_name = model_name
        self._device = device
        self._model = None  # lazy-loaded on first use

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            print(f"Loading cross-encoder reranker: {self.model_name}")
            # device=None lets sentence-transformers auto-select CUDA if available
            self._model = CrossEncoder(self.model_name, device=self._device)
        return self._model

    def rerank(self, query: str, documents: list, top_k: int):
        """Score each document against the query and return the top_k.

        Args:
            query:     the user question.
            documents: a list of LangChain Document objects (Stage 1 candidates).
            top_k:     how many documents to keep after reranking.

        Returns:
            A list of (Document, score) tuples sorted by descending relevance,
            truncated to top_k. The score is the raw cross-encoder logit.
        """
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]
