# Re-generating the exact English README content to ensure file persistence and retrieve a fresh file tag.
markdown_content = """# 🏛️ Project Alexandria (V1.1)
## Enterprise-Grade Local RAG Knowledge Base for Complex Technical Documentation

**Subject:** Capstone Project (Bachelor of Science in Computer Science)  
**Architecture:** 100% Local, Privacy-Compliant Retrieval-Augmented Generation (RAG)  
**Hardware Target:** NVIDIA CUDA Accelerated (Optimized for RTX GPUs with Tensor Cores)

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Hardware & Software Requirements](#hardware--software-requirements)
5. [Installation Guide](#installation-guide)
6. [Execution Pipeline](#execution-pipeline)
7. [State Management & Resilience](#state-management--resilience)

---

## 🎯 Project Overview

**Alexandria** is an Enterprise-grade **Retrieval-Augmented Generation (RAG)** infrastructure designed to operate entirely on local hardware. The project addresses a critical corporate challenge: safely extracting and querying knowledge from complex, unstructured technical documentation (e.g., user manuals, POS/TCPOS system configurations, industrial protocols) while guaranteeing absolute **Data Sovereignty** and zero communication with external Cloud APIs.

The architecture engineers a complete end-to-end pipeline, ranging from preliminary *Data Sanitization* to extraction via *AI Vision* models, indexing into a *Vector Database*, and final response generation through a high-performance, quantized *Large Language Model (LLM)*.

---

## ✨ Key Features

- **Fault-Tolerant Data Ingestion (Idempotent Batch Processor):** Page-by-page PDF processing with persistent state saving (*Checkpointing*). In the event of an interruption or manual halt, the system seamlessly resumes from the last successfully processed page.
- **Vision-Based PDF Parsing (Marker/Surya OCR):** Advanced layout-aware PDF-to-Markdown conversion capable of surgically extracting tables, hierarchical formatting, and structured text blocks, overcoming the severe limitations of traditional naive string-scraping.
- **Markdown-Aware Hierarchical Chunking:** A semantic chunking strategy that respects Markdown structural tags (Headers `#`, `##`, `###`), preserving table integrity and preventing context fragmentation within the Vector DB.
- **Enterprise Source Traceability:** Every text chunk ingested into the vector database is tagged with immutable metadata containing the original PDF page number, allowing the LLM to cite verifiable sources.
- **High-Efficiency Inference Server (llama.cpp):** Response generation is delegated to the native C++ `llama.cpp` engine with full GPU layer offloading on Tensor Cores, configured with strict temperature constraints to eliminate model hallucinations.

---

## 🧠 System Architecture

Alexandria's pipeline is divided into 4 sequential, decoupled software modules: 
[ Raw PDF Document ]
│
▼
┌─────────────────────────────┐
│ 1. DATA SANITIZER           │ ──> Purges indices, appendices, and blank pages (PyMuPDF)
└─────────────────────────────┘
│
▼
┌─────────────────────────────┐
│ 2. BATCH INGESTION (OCR)    │ ──> Isolates and converts pages to Markdown (Marker + Checkpoint)
└─────────────────────────────┘
│
▼
┌─────────────────────────────┐
│ 3. CHUNKING & VECTOR DB     │ ──> Splits by Headers & computes Embeddings (ChromaDB + HuggingFace)
└─────────────────────────────┘
│
▼
┌─────────────────────────────┐
│ 4. QUERY ENGINE (RAG)       │ <──> Queries the local LLM via HTTP API (llama.cpp Server)
└─────────────────────────────┘

---

## 💻 Hardware & Software Requirements

### Minimum / Recommended Hardware
- **CPU:** x86_64 Multi-core (Intel i7/i9 or AMD Ryzen 7/9)
- **GPU:** Dedicated NVIDIA (Ampere/Ada Lovelace architecture or higher, e.g., RTX 3090, RTX 4070 Ti, RTX 5070 Ti)
- **VRAM:** Minimum 12 GB dedicated (16GB+ recommended for concurrent Embedding & LLM execution)

### Software Environment
- Windows 10/11 (PowerShell) or Linux (Ubuntu 22.04+)
- Python 3.10 / 3.11 (Isolated installation via `.venv`)
- NVIDIA CUDA Toolkit (Compatible with host system drivers)

---

## 🛠️ Installation Guide

In compliance with Systems Engineering best practices, the computing environment is strictly isolated. PyTorch (the accelerated neural backend) must be installed by explicitly pointing to the CUDA binaries matched to the host hardware.

### 1. Virtual Environment Initialization
```powershell
# Clone or navigate to the project root directory
cd C:\\Code\\alexandria\\AlexandriaV1.1
```

# Create and activate the virtual environment
```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

### 2. Surgical PyTorch Installation (CUDA Accelerated)

Install the neural core configured for CUDA drivers (Supported stable release cu124):

```powershell
pip install torch torchvision torchaudio --extra-index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
```

### 3. Standard Dependencies Installation (Requirements)
Install the remaining infrastructure dependencies and connectors:
```powershell
pip install -r requirements.txt
```

## 🚀 Execution Pipeline
### Phase 1: Data Pre-processing & Sanitization
Removes "vector noise" (e.g., analytical indices, covers, legal pages) by extracting only the core technical page ranges.