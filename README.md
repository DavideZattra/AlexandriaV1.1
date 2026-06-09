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
- **Agentic RAG Router (LangGraph):** A stateful LangGraph agent classifies each query as `manual_search` or `chitchat`, routing it to either the ChromaDB retriever or a direct LLM response path.
- **High-Efficiency Local Inference (llama.cpp):** Response generation is delegated to the native C++ `llama.cpp` engine with full GPU layer offloading on Tensor Cores, configured with strict temperature constraints to minimize hallucinations.

---

## System Architecture

Alexandria's pipeline is divided into 6 sequential, decoupled software modules:

```
[ Raw PDF Document ]
        │
        ▼
┌─────────────────────────────┐
│  0. DATA SANITIZER          │  ──> Strips covers, indices, legal pages (PyMuPDF)
│  sanitize_pdfs.py           │      Outputs: Manuals/Sanitized/tcpos_manual.pdf
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  1. MACRO-CHAPTER SPLITTER  │  ──> Divides PDF into named chapter files (PyMuPDF)
│  alexandria_chapter_        │      Configurable page-offset alignment
│  splitter.py                │      Outputs: alexandria_macro_chapters/
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  2. BATCH INGESTION (OCR)   │  ──> Isolates pages → Markdown via Marker + Surya
│  batch_ingestion.py         │      Fault-tolerant with checkpoint resume
│                             │      Outputs: alexandria_knowledge_base/page_XXXX/
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  3. CHUNKING                │  ──> Splits Markdown by headers + size
│  alexandria_chunking.py     │      Injects page & document metadata per chunk
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  4. VECTOR DATABASE         │  ──> Computes embeddings, persists to ChromaDB
│  alexandria_vector_db.py    │      Model: all-MiniLM-L6-v2 (local, ~100MB)
│                             │      Outputs: alexandria_db/
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  5. RAG QUERY ENGINE        │  <──> LangGraph router → ChromaDB retriever
│  router.py                  │       → llama.cpp LLM server (localhost:8080)
└─────────────────────────────┘
```

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

Run the scripts **in order** for a full end-to-end build of the knowledge base.

### Phase 0 — Data Sanitization (`sanitize_pdfs.py`)
Strips noise pages (covers, legal notices, analytical indices) by extracting only the core technical page ranges.

```powershell
python sanitize_pdfs.py
```

Configure `PAGES_TO_KEEP` in the script to define which page ranges to retain (0-indexed).  
**Output:** `Manuals/Sanitized/tcpos_manual.pdf`

---

### Phase 1 — Macro-Chapter Splitting (`alexandria_chapter_splitter.py`)
Divides the sanitized PDF into individual chapter PDFs using a pre-defined chapter map and page-offset alignment.

```powershell
python alexandria_chapter_splitter.py
```

Configure `CHAPTERS` (name, printed start/end) and `PAGE_OFFSET` (the difference between the printed page number and the physical PDF page) in the script.  
**Output:** `alexandria_macro_chapters/01_General_Concepts_...pdf`, `02_Admin_module.pdf`, etc.

---

### Phase 2 — Batch OCR Ingestion (`batch_ingestion.py`)
Processes the sanitized PDF page-by-page through Marker + Surya OCR, converting each page to a structured Markdown file.

```powershell
python batch_ingestion.py
```

- Automatically resumes from the last checkpoint if interrupted.
- Prompts every 120 minutes to allow GPU cooling pauses.

**Output:** `alexandria_knowledge_base/page_XXXX/<page_name>.md`

To restart from scratch, delete `alexandria_checkpoint.txt`.

---

### Phase 3 — Chunking & Vector DB Build (`alexandria_vector_db.py`)
Reads all generated Markdown files, splits them semantically by Markdown headers, and ingests the chunks into a persistent ChromaDB vector store.

```powershell
python alexandria_vector_db.py
```

This script calls `alexandria_chunking.py` internally. Embeddings are computed locally using `sentence-transformers/all-MiniLM-L6-v2`.  
**Output:** `alexandria_db/` (ChromaDB persistent store)

---

### Phase 4 — Query the Knowledge Base (`router.py`)
The LangGraph-based RAG agent classifies the user query and routes it to the appropriate node:
- `manual_search` → retrieves context from ChromaDB → generates answer via llama.cpp
- `chitchat` → responds directly via llama.cpp without retrieval

```python
from router import app
from langchain_core.messages import HumanMessage

result = app.invoke({"question": "How do I configure a new outlet in TCPOS?"})
print(result["generation"])
```

Requires the `llama.cpp` server running on `http://localhost:8080`.

---

## State Management & Resilience

The batch ingestion system (`batch_ingestion.py`) is designed for long-running GPU workloads on large documents (1000+ pages):

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
├── sanitize_pdfs.py              # Phase 0: PDF noise removal
├── alexandria_chapter_splitter.py # Phase 1: Chapter-level PDF splitting
├── batch_ingestion.py            # Phase 2: Page-by-page OCR ingestion (Marker)
├── alexandria_chunking.py        # Phase 3a: Markdown semantic chunking
├── alexandria_vector_db.py       # Phase 3b: ChromaDB vector store builder
├── router.py                     # Phase 4: LangGraph RAG query router
├── ingest.py                     # (Legacy) HTML/CHM ingestion prototype
├── vision_extractor.py           # Vision LLM extraction utility
├── cleaner.py                    # Auxiliary text cleaning utilities
├── pdf_tester.py                 # PDF validation/debug tool
├── alexandria_checkpoint.txt     # Ingestion resume state
├── alexandria_knowledge_base/    # OCR output (Markdown per page)
├── alexandria_macro_chapters/    # Chapter-split PDFs
├── alexandria_db/                # ChromaDB persistent vector store
├── Manuals/                      # Source PDFs (raw + sanitized)
├── Docs/                         # Project documentation & requirements
└── .venv/                        # Isolated Python environment
```
