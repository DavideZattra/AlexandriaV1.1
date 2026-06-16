"""Contiguous Page Context Expansion (post-retrieval).

A single retrieved chunk is often only a fragment of a larger idea — a procedure
may begin at the bottom of one page and finish on the next, or a table's header
and rows may land in different chunks. After reranking has selected the most
relevant chunks, this module expands each one with the chunks from its
neighbouring pages (within +/- window pages) and stitches everything back
together in natural reading (page) order.

It fetches neighbours directly from ChromaDB by metadata, using an `$in` filter
on the exact target page numbers. This matters: using a min/max range would
pull every page between two far-apart hits (e.g. page 12 and page 450), so we
enumerate the precise pages instead.
"""

from collections import defaultdict

from config import CONTEXT_WINDOW


def _as_documents(ranked):
    """Accept either a list of Documents or (Document, score) tuples."""
    docs = []
    for item in ranked:
        docs.append(item[0] if isinstance(item, tuple) else item)
    return docs


def _coerce_page(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def expand_contiguous_pages(vector_db, ranked, window: int = CONTEXT_WINDOW):
    """Expand reranked chunks with their neighbouring pages.

    Args:
        vector_db: the Chroma vector store to fetch neighbouring chunks from.
        ranked:    reranked results (list of Documents or (Document, score) tuples).
        window:    number of pages to include on each side of each retrieved page.

    Returns:
        A list of formatted context blocks, one per page, ordered by document
        name then ascending page number. Each block keeps the
        "[Source: <doc>, Page <n>]" tag so downstream citation parsing still works.
    """
    docs = _as_documents(ranked)
    if not docs:
        return []

    # 1. Collect the exact set of target pages per document (with the window).
    targets = defaultdict(set)
    for doc in docs:
        name = doc.metadata.get("document_name", "manual")
        page = _coerce_page(doc.metadata.get("source_page"))
        if page is None:
            continue
        for p in range(page - window, page + window + 1):
            if p >= 1:  # page numbers are 1-based; never request page 0 or negative
                targets[name].add(p)

    if not targets:
        return []

    # 2. Fetch all chunks for those exact pages and group them by (doc, page).
    page_chunks = defaultdict(list)  # (document_name, page) -> [chunk_text, ...]
    for name, pages in targets.items():
        page_list = sorted(pages)
        where = {"$and": [
            {"document_name": {"$eq": name}},
            {"source_page": {"$in": page_list}},
        ]}
        result = vector_db.get(where=where)
        contents = result.get("documents", []) or []
        metadatas = result.get("metadatas", []) or []
        for content, meta in zip(contents, metadatas):
            page = _coerce_page(meta.get("source_page"))
            if page is not None:
                page_chunks[(name, page)].append(content)

    # 3. Stitch chunks into one block per page, ordered by document then page.
    blocks = []
    for (name, page) in sorted(page_chunks.keys()):
        body = "\n".join(page_chunks[(name, page)])
        blocks.append(f"[Source: {name}, Page {page}]\n{body}")

    return blocks
