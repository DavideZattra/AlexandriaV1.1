#!/usr/bin/env python
"""Entrypoint: build the ChromaDB vector database from the chunked knowledge base.

Accepts --profile minilm|bge|all (see alexandria.ingestion.vector_db).
"""

from alexandria.ingestion.vector_db import main

if __name__ == "__main__":
    main()
