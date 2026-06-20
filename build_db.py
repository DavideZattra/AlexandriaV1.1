#!/usr/bin/env python
"""Entrypoint: build the ChromaDB vector database from the chunked knowledge base."""

from alexandria.ingestion.vector_db import build_vector_db

if __name__ == "__main__":
    build_vector_db()
