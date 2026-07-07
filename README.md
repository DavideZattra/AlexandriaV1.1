# Project Alexandria
## Enterprise-Grade Local RAG Knowledge Base for Complex Technical Documentation

**Subject:** Capstone Project — Bachelor of Science in Computer Science  
**Architecture:** 100% Local, Privacy-Compliant Retrieval-Augmented Generation (RAG)  
**Hardware Target:** NVIDIA CUDA Accelerated (Optimized for RTX GPUs with Tensor Cores)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Hardware & Software Requirements](#hardware--software-requirements)
5. [Installation Guide](#installation-guide)
6. [Execution Pipeline](#execution-pipeline)
7. [State Management & Resilience](#state-management--resilience)
8. [Project Structure](#project-structure)

---

## Project Overview

**Alexandria** is an enterprise-grade **Retrieval-Augmented Generation (RAG)** infrastructure designed to operate entirely on local hardware. The project addresses a critical corporate challenge: safely extracting and querying knowledge from complex, unstructured technical documentation (e.g., user manuals, POS/TCPOS system configurations, industrial protocols) while guaranteeing absolute **Data Sovereignty** and zero communication with external Cloud APIs.

The architecture engineers a complete end-to-end pipeline, from preliminary *Data Sanitization* and *Macro-Chapter Splitting* to AI-powered *Vision OCR extraction*, *semantic chunking* into a *Vector Database*, and final response generation through a high-performance, quantized *Large Language Model (LLM)*.

---

## Key Features

- **Fault-Tolerant Batch Ingestion (Idempotent):** Page-by-page PDF processing with persistent state saving via checkpointing. In the event of an interruption or manual halt, the system seamlessly resumes from the last successfully processed page.
- **Macro-Chapter Pre-Splitter:** Before OCR ingestion, the raw PDF is surgically divided into named chapter files using a configurable page-offset alignment map. This enables modular, chapter-scoped processing.
- **Vision-Based PDF Parsing (Marker + Surya OCR):** Advanced layout-aware PDF-to-Markdown conversion capable of extracting tables, hierarchical formatting, and structured text blocks — overcoming severe limitations of traditional text-scraping.
- **Markdown-Aware Hierarchical Chunking:** A semantic chunking strategy that respects Markdown structural tags (`#`, `##`, `###`), preserving table integrity and preventing context fragmentation within the vector DB.
- **Enterprise Source Traceability:** Every chunk ingested into the vector database is tagged with immutable metadata (source page number, document name), allowing the LLM to cite verifiable sources.
- **Dual Embedding Profiles:** Two embedding models coexist in separate ChromaDB stores for A/B accuracy comparison — `BAAI/bge-large-en-v1.5` (1024-dim, the default, with its asymmetric query-instruction prefix applied at query time) and `all-MiniLM-L6-v2` (384-dim). Selectable via `--profile` / `--embedder` flags.
- **Two-Stage Cross-Encoder Reranking:** A wide bi-encoder candidate pool (ChromaDB) is re-scored by a local cross-encoder (`ms-marco-MiniLM-L-6-v2`), keeping only the most relevant chunks for generation.
- **Contiguous Page Context Expansion:** Reranked chunks are expanded with their neighbouring pages (±1 page) and stitched in reading order, so procedures and tables spanning page breaks stay intact.
- **Agentic RAG (LangGraph):** A stateful cyclic graph, not a linear pipeline. Queries are classified (`manual_search` / `chitchat`); retrieved context is **graded** (CRAG) and, if insufficient, the query is **rewritten** and retrieval retried; generated answers pass a **Self-RAG grounding check** that triggers a strict regeneration or an explicit warning caveat when claims are not supported by the retrieved pages.
- **Grammar-Constrained Verdicts (GBNF):** All agentic yes/no decisions use llama.cpp grammar-constrained decoding — the sampler can only emit tokens matching a JSON grammar, making malformed verdicts impossible and skipping reasoning-model preambles (~0.2 s per verdict). Layered fallbacks (lenient parsing, safe defaults) mean a verdict can never crash the graph.
- **High-Efficiency Local Inference (llama.cpp):** Response generation is delegated to the native C++ `llama.cpp` engine with full GPU layer offloading on Tensor Cores, configured with strict temperature constraints to minimize hallucinations.

---

## System Architecture

Alexandria's pipeline is divided into 6 sequential, decoupled software modules:

```
[ Raw PDF Document ]
        │
        ▼
┌──────────────────────────────────────┐
│  0. DATA SANITIZER                    │ ─> Strips covers, indices, legal pages (PyMuPDF)
│  alexandria/ingestion/sanitize_pdfs.py│    Outputs: Manuals/Sanitized/tcpos_manual.pdf
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│  1. MACRO-CHAPTER SPLITTER            │ ─> Divides PDF into named chapter files (PyMuPDF)
│  alexandria/ingestion/                │    Configurable page-offset alignment
│      chapter_splitter.py              │    Outputs: alexandria_macro_chapters/
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│  2. BATCH INGESTION (OCR)             │ ─> Isolates pages → Markdown via Marker + Surya
│  alexandria/ingestion/batch_ingestion.py│  Fault-tolerant with checkpoint resume
│                                       │    Outputs: alexandria_knowledge_base/page_XXXX/
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│  3. CHUNKING                          │ ─> Splits Markdown by headers + size
│  alexandria/ingestion/chunking.py     │    Injects page & document metadata per chunk
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│  4. VECTOR DATABASE                   │ ─> Computes embeddings, persists to ChromaDB
│  alexandria/ingestion/vector_db.py    │    Profiles: bge-large-en-v1.5 (default) /
│  (+ embeddings.py profile registry)   │    all-MiniLM-L6-v2 — one DB directory each
│                                       │    Outputs: alexandria_db_bge/, alexandria_db/
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│  5. AGENTIC RAG QUERY ENGINE          │ <─> LangGraph cyclic graph (see below)
│  alexandria/agent/graph.py            │     retrieve → rerank → expand → grade
│  (+ retrieval/reranker.py,            │     → generate → grounding check
│     retrieval/context_expansion.py,   │     → llama.cpp LLM server (localhost:8080)
│     core/structured.py GBNF verdicts) │
└──────────────────────────────────────┘
```

### Agentic Query Flow

The query engine is a **cyclic** LangGraph, not a fixed pipeline. Two self-correction
loops (CRAG document grading + query rewriting, and a Self-RAG grounding check) let
the system retry weak retrievals and refuse to return unsupported claims:

```
[router] ─(chitchat)──────────────────────────────────────────> END
    │
    └(manual_search)
         ▼
     [retrieve] <────────────────┐          retrieve = similarity top-20
         ▼                       │                     → cross-encoder rerank top-5
  [grade_documents]              │                     → page expansion ±1
         ├─ yes ──────────────┐  │
         └─ no ──> [rewrite_query]          (≤ 2 rewrites, then best-effort)
                              ▼
                        [generate] <──────┐
                              ▼           │
                     [check_grounding]    │
                        ├─ grounded ──────┼──> END
                        ├─ not grounded ──┘   (1 strict regeneration)
                        └─ still failing ───> END with explicit warning caveat
```

Every branch decision is a GBNF grammar-constrained JSON verdict (`core/structured.py`):
malformed output is impossible at the sampler level, with lenient parsing and safe
defaults as fallback layers. Loop caps (`RETRIEVAL_MAX_RETRIES`, `GROUNDING_MAX_RETRIES`
in `config.py`) bound the extra LLM calls.

---

## Hardware & Software Requirements

### Minimum / Recommended Hardware
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | x86_64 Multi-core (i7 / Ryzen 7) | Intel i9 / Ryzen 9 |
| GPU | NVIDIA Ampere+ (e.g., RTX 3090) | RTX 4070 Ti / RTX 5070 Ti |
| VRAM | 12 GB dedicated | 16 GB+ |
| RAM | 32 GB | 64 GB |
| Storage | 50 GB free (SSD) | 100 GB+ NVMe SSD |

### Software Environment
- Windows 10/11 (PowerShell) or Linux (Ubuntu 22.04+)
- Python 3.10 / 3.11 (isolated `.venv`)
- NVIDIA CUDA Toolkit (matched to host driver version)
- `llama.cpp` server binary (for LLM inference)

---

## Installation Guide

### 1. Clone and navigate to the project
```powershell
cd C:\Code\alexandria\AlexandriaV1.1
```

### 2. Create and activate the virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install PyTorch with CUDA support
Install the neural backend configured for CUDA 12.4 drivers:
```powershell
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
```

### 4. Install remaining dependencies
```powershell
pip install -r Docs/requirements.txt
```

### 5. Start the llama.cpp inference server
Before running queries, start the local LLM server (outside this repo):
```powershell
.\llama-server.exe -m <path_to_your_model.gguf> --gpu-layers 99 --port 8080
```

---

## Execution Pipeline

Run the stages **in order** for a full end-to-end build of the knowledge base. All
commands are run from the project root. The offline ingestion stages are invoked as
package modules (`python -m alexandria.ingestion.*`); the common runtime tasks have
thin entrypoint scripts at the root (`build_db.py`, `run_chat.py`, `evaluate.py`).

### Phase 0 — Data Sanitization
Strips noise pages (covers, legal notices, analytical indices) by extracting only the core technical page ranges.

```powershell
python -m alexandria.ingestion.sanitize_pdfs
```

Configure `PAGES_TO_KEEP` in `alexandria/ingestion/sanitize_pdfs.py` to define which page ranges to retain (0-indexed).  
**Output:** `Manuals/Sanitized/tcpos_manual.pdf`

---

### Phase 1 — Macro-Chapter Splitting
Divides the sanitized PDF into individual chapter PDFs using a pre-defined chapter map and page-offset alignment.

```powershell
python -m alexandria.ingestion.chapter_splitter
```

Configure `CHAPTERS` (name, printed start/end) and `PAGE_OFFSET` (the difference between the printed page number and the physical PDF page) in `alexandria/ingestion/chapter_splitter.py`.  
**Output:** `alexandria_macro_chapters/01_General_Concepts_...pdf`, `02_Admin_module.pdf`, etc.

---

### Phase 2 — Batch OCR Ingestion
Processes the sanitized PDF page-by-page through Marker + Surya OCR, converting each page to a structured Markdown file.

```powershell
python -m alexandria.ingestion.batch_ingestion
```

- Automatically resumes from the last checkpoint if interrupted.
- Prompts every 120 minutes to allow GPU cooling pauses.

**Output:** `alexandria_knowledge_base/page_XXXX/<page_name>.md`

To restart from scratch, delete `alexandria_checkpoint.txt`.

---

### Phase 3 — Chunking & Vector DB Build
Reads all generated Markdown files, splits them semantically by Markdown headers, and ingests the chunks into a persistent ChromaDB vector store.

```powershell
python build_db.py                    # default profile (bge)
python build_db.py --profile minilm   # or: bge | all (chunks once, builds every profile)
```

This invokes `alexandria/ingestion/vector_db.py`, which calls `alexandria/ingestion/chunking.py` internally. Each embedding profile (`bge` = `BAAI/bge-large-en-v1.5`, `minilm` = `all-MiniLM-L6-v2`) gets its own database directory, since their vector dimensions differ (1024 vs 384).  
**Output:** `alexandria_db_bge/` and/or `alexandria_db/` (ChromaDB persistent stores)

---

### Phase 4 — Query the Knowledge Base
The agentic LangGraph RAG engine (`alexandria/agent/graph.py`) runs each query through the cyclic graph shown in [Agentic Query Flow](#agentic-query-flow):
- `manual_search` → retrieve (rerank + page expansion) → **grade context** (rewrite query and retry if insufficient) → generate → **grounding check** (strict regeneration or explicit caveat if unsupported)
- `chitchat` → respond directly via llama.cpp without retrieval

Every turn is logged as JSONL (question, retrieved context, loop verdicts, latency) under `logs/` for post-mortem analysis.

Interactive chat:

```powershell
python run_chat.py
```

Or programmatically:

```python
from alexandria.agent.graph import app

result = app.invoke({"question": "How do I configure a new outlet in TCPOS?"})
print(result["generation"])
print(result["grade"], result["grounded"])   # loop verdicts
```

Requires the `llama.cpp` server running on `http://localhost:8080`.

---

### Evaluation
Compare pipeline configurations across the fixed test query set, producing Markdown + JSON reports (including per-answer agentic loop verdicts) under `Docs/Evaluation_Results/`:

| Mode | Pipeline |
|------|----------|
| `direct_llm` | No retrieval (control) |
| `baseline` | ChromaDB similarity top-5 only |
| `reranked` | top-20 → cross-encoder rerank → top-5 |
| `expanded` | reranked + contiguous page expansion |
| `agentic` | Full LangGraph agent with CRAG + grounding loops |

```powershell
python evaluate.py --mode all --queries all            # full ablation ladder
python evaluate.py --mode expanded --embedder both     # A/B the two embedding profiles
```

The `--embedder minilm|bge|both` flag runs the retrieval modes against each embedding profile's database (`agentic` and `direct_llm` run once — the agent owns its own store).

---

## State Management & Resilience

The batch ingestion system (`alexandria/ingestion/batch_ingestion.py`) is designed for long-running GPU workloads on large documents (1000+ pages):

| Mechanism | Description |
|-----------|-------------|
| **Checkpoint file** | `alexandria_checkpoint.txt` stores the last successfully processed page index. |
| **Resume on restart** | On next run, the script reads the checkpoint and skips already-processed pages. |
| **Manual pause** | Every 120 minutes the script prompts `y/n` to continue, allowing GPU cooling. |
| **Error isolation** | If Marker fails on a single page, the script halts with the checkpoint preserved — no data is lost. |

---

## Project Structure

```
AlexandriaV1.1/
├── run_chat.py                       # Entrypoint: interactive RAG chat
├── build_db.py                       # Entrypoint: build the ChromaDB vector store
├── evaluate.py                       # Entrypoint: run the evaluation harness
│
├── alexandria/                       # Main package
│   ├── paths.py                      # Project-root-anchored data directory paths
│   ├── config.py                     # Shared config (embedding profiles, rerank/expansion params, loop caps)
│   ├── embeddings.py                 # Embedding-profile registry loader + query-prefix wrapper
│   ├── core/
│   │   ├── conversation_logger.py    # JSONL per-session logging (incl. agentic loop verdicts)
│   │   └── structured.py             # GBNF grammar-constrained JSON verdicts + fallbacks
│   ├── ingestion/                    # Offline knowledge-base build pipeline
│   │   ├── sanitize_pdfs.py          # Phase 0: PDF noise removal
│   │   ├── chapter_splitter.py       # Phase 1: Chapter-level PDF splitting
│   │   ├── batch_ingestion.py        # Phase 2: Page-by-page OCR ingestion (Marker)
│   │   ├── chunking.py               # Phase 3a: Markdown semantic chunking
│   │   ├── vector_db.py              # Phase 3b: ChromaDB vector store builder (--profile)
│   │   ├── vision_extractor.py       # Vision-LLM (Qwen2.5-VL) extraction utility
│   │   └── legacy/                   # Early prototypes (ingest.py, cleaner.py, pdf_tester.py)
│   ├── retrieval/                    # Query-time retrieval building blocks
│   │   ├── reranker.py               # Cross-encoder reranker (Stage 2)
│   │   └── context_expansion.py      # Contiguous page context expansion (Stage 3)
│   ├── agent/
│   │   └── graph.py                  # Phase 4: agentic LangGraph (CRAG + grounding loops) + chat loop
│   └── eval/
│       └── evaluate_queries.py       # Evaluation harness (direct/baseline/reranked/expanded/agentic)
│
├── alexandria_checkpoint.txt         # Ingestion resume state
├── alexandria_knowledge_base/        # OCR output (Markdown per page)
├── alexandria_macro_chapters/        # Chapter-split PDFs
├── alexandria_db_bge/                # ChromaDB store — bge profile (default)
├── alexandria_db/                    # ChromaDB store — minilm profile
├── logs/                             # JSONL conversation logs (per session)
├── Manuals/                          # Source PDFs (raw + sanitized)
├── Tests/                            # Standalone test scripts (incl. test_structured.py)
├── Docs/                             # Project documentation & requirements
└── .venv/                            # Isolated Python environment
```
