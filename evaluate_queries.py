#!/usr/bin/env python
"""
Alexandria RAG Evaluation Runner (Baseline Edition)
Runs the 22 queries in Docs/Test_Queries.txt through your current baseline RAG setup
and direct LLM execution to evaluate performance, accuracy, and latency.
"""

import os
import sys
import time
import json
import argparse
import re
from datetime import datetime

# Import core RAG components and parameters from router
try:
    from router import llm, reranker
    from config import RERANK_CANDIDATES, RERANK_TOP_K, CONTEXT_WINDOW, EMBEDDING_PROFILES, DEFAULT_PROFILE
    from context_expansion import expand_contiguous_pages
    from embeddings import load_vector_db
except ImportError:
    print("Error: Could not import components from router.py. Make sure this script is run from the workspace root.")
    sys.exit(1)

from langchain_core.messages import HumanMessage, SystemMessage

OUTPUT_DIR = "./Docs/Evaluation_Results"
MK_DIR = "/Md_Reports"
LOG_DIR = "/Log_Files"

def load_queries(file_path):
    """Load and parse test queries from Test_Queries.txt."""
    queries = []
    if not os.path.exists(file_path):
        print(f"Error: Queries file not found at {file_path}")
        return queries

    encodings = ['utf-8', 'latin-1', 'cp1252', 'utf-16']
    content = None
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        print(f"Error: Failed to read {file_path} with any common encoding.")
        return queries

    lines = content.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Split on the first dot (e.g. "1. What is...")
        match = re.match(r"^(\d+)\.\s*(.*)$", line)
        if match:
            num = int(match.group(1))
            text = match.group(2).strip()
            queries.append((num, text))
        else:
            # Fallback if line format is different
            queries.append((len(queries) + 1, line))
            
    return queries

def retrieve_baseline(question: str, vdb):
    """Stage-1 only: pure bi-encoder similarity, top RERANK_TOP_K chunks.

    Uses the same final chunk count as the reranked mode so the two are an
    apples-to-apples comparison (same number of chunks fed to the LLM; the only
    difference is whether they were reranked)."""
    print(f"--- Retrieving from ChromaDB (top {RERANK_TOP_K}, no rerank) ---")
    results = vdb.similarity_search(question, k=RERANK_TOP_K)
    retrieved_docs = []
    for doc in results:
        page = doc.metadata.get("source_page", "unknown")
        document_name = doc.metadata.get("document_name", "manual")
        retrieved_docs.append(f"[Source: {document_name}, Page {page}]\n{doc.page_content}")
    return retrieved_docs

def retrieve_reranked(question: str, vdb):
    """Two-stage retrieval: bi-encoder candidate pool + cross-encoder rerank.

    Stage 1 retrieves RERANK_CANDIDATES via ChromaDB, Stage 2 reranks them with
    the cross-encoder and keeps the top RERANK_TOP_K."""
    print(f"--- Retrieving {RERANK_CANDIDATES} candidates, reranking to top {RERANK_TOP_K} ---")
    candidates = vdb.similarity_search(question, k=RERANK_CANDIDATES)
    if not candidates:
        return []
    ranked = reranker.rerank(question, candidates, top_k=RERANK_TOP_K)
    retrieved_docs = []
    for doc, score in ranked:
        page = doc.metadata.get("source_page", "unknown")
        document_name = doc.metadata.get("document_name", "manual")
        retrieved_docs.append(f"[Source: {document_name}, Page {page}]\n{doc.page_content}")
    return retrieved_docs

def retrieve_expanded(question: str, vdb):
    """Full pipeline: bi-encoder + cross-encoder rerank + contiguous page expansion.

    Stage 1 retrieves RERANK_CANDIDATES, Stage 2 reranks to RERANK_TOP_K, Stage 3
    expands each surviving chunk with its neighbouring pages (+/- CONTEXT_WINDOW)."""
    print(f"--- Retrieving {RERANK_CANDIDATES}, rerank to {RERANK_TOP_K}, expand +/- {CONTEXT_WINDOW} page(s) ---")
    candidates = vdb.similarity_search(question, k=RERANK_CANDIDATES)
    if not candidates:
        return []
    ranked = reranker.rerank(question, candidates, top_k=RERANK_TOP_K)
    return expand_contiguous_pages(vdb, ranked, window=CONTEXT_WINDOW)

def generate_answer(question: str, retrieved_docs: list) -> str:
    """Send retrieved context + question to the local LLM."""
    if not retrieved_docs:
        return "I could not find any relevant information in the manual to answer this question."
        
    context = "\n\n".join(retrieved_docs)
    prompt = [
        SystemMessage(content=(
            "Answer the user's question based ONLY on the context below. "
            "Each context block is tagged with its source document and page number. "
            "Cite the page number(s) you used in your answer. "
            "If the context does not contain the answer, say so explicitly.\n\n"
            f"CONTEXT:\n{context}"
        )),
        HumanMessage(content=question)
    ]
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Error during generation: {e}"

def generate_direct(question: str) -> str:
    """Send query directly to the local LLM without context."""
    prompt = [
        SystemMessage(content="You are a helpful assistant. Answer the user question as accurately as possible."),
        HumanMessage(content=question)
    ]
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Error during direct generation: {e}"

def run_mode(mode, question, vdb=None):
    """Execute a single query with the selected mode configuration.

    vdb is the embedding profile's vector store; required for retrieval modes,
    ignored for direct_llm (which is embedder-independent)."""
    start_time = time.perf_counter()
    retrieved_docs = []

    if mode == "baseline":
        retrieved_docs = retrieve_baseline(question, vdb)
        generation = generate_answer(question, retrieved_docs)
    elif mode == "reranked":
        retrieved_docs = retrieve_reranked(question, vdb)
        generation = generate_answer(question, retrieved_docs)
    elif mode == "expanded":
        retrieved_docs = retrieve_expanded(question, vdb)
        generation = generate_answer(question, retrieved_docs)
    elif mode == "direct_llm":
        generation = generate_direct(question)
    else:
        raise ValueError(f"Unknown mode: {mode}")
        
    latency = time.perf_counter() - start_time
    
    # Extract source page citations from retrieved docs
    citations = []
    for doc in retrieved_docs:
        match = re.search(r"Page (\d+)", doc)
        if match:
            citations.append(int(match.group(1)))
    citations = sorted(list(set(citations)))
    
    return {
        "generation": generation,
        "latency_s": latency,
        "citations": citations,
        "num_sources": len(retrieved_docs)
    }

def parse_queries_arg(queries_str, loaded_queries):
    """Parse comma-separated query list or select 'all'."""
    if not queries_str or queries_str.lower() == "all":
        return loaded_queries
        
    requested_ids = []
    for part in queries_str.split(","):
        part = part.strip()
        if "-" in part:
            # Handle range (e.g. 1-5)
            try:
                start, end = map(int, part.split("-"))
                requested_ids.extend(range(start, end + 1))
            except ValueError:
                pass
        else:
            try:
                requested_ids.append(int(part))
            except ValueError:
                pass
                
    selected = [q for q in loaded_queries if q[0] in requested_ids]
    if not selected:
        print(f"Warning: No valid queries selected matching '{queries_str}'. Defaulting to all.")
        return loaded_queries
    return selected

def write_markdown_report(results, modes, timestamp, descriptions=None):
    """Write comparison results in a beautiful Markdown report.

    `modes` here are run labels (e.g. 'baseline[bge]') and `descriptions` maps
    each label to a human-readable summary."""
    descriptions = descriptions or {}
    os.makedirs(OUTPUT_DIR + MK_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR + MK_DIR, f"report_baseline_{timestamp}.md")

    # Compute aggregates
    avg_latency = {}
    for mode in modes:
        latencies = [q["runs"][mode]["latency_s"] for q in results if mode in q["runs"]]
        avg_latency[mode] = sum(latencies) / len(latencies) if latencies else 0.0

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Alexandria RAG Baseline Evaluation Report\n\n")
        f.write(f"*Generated on:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"This report compares the performance and answer quality of your current baseline RAG setup against direct LLM response.\n\n")

        f.write(f"## Configuration Performance Summary\n\n")
        f.write(f"| Mode | Avg Latency (s) | Queries Run | Description |\n")
        f.write(f"| :--- | :--- | :--- | :--- |\n")
        for mode in modes:
            f.write(f"| **{mode}** | {avg_latency[mode]:.2f}s | {len(results)} | {descriptions.get(mode, '')} |\n")
            
        f.write(f"\n---\n\n## Detailed Query Comparisons\n\n")
        
        for q in results:
            f.write(f"### Query {q['id']}: {q['question']}\n\n")
            
            # Print comparative table
            f.write(f"| Mode | Latency | Sources Cited | Num Context Chunks |\n")
            f.write(f"| :--- | :--- | :--- | :--- |\n")
            for mode in modes:
                run = q["runs"][mode]
                citations_str = ", ".join(map(str, run["citations"])) if run["citations"] else "None"
                f.write(f"| **{mode}** | {run['latency_s']:.2f}s | Page(s) {citations_str} | {run['num_sources']} |\n")
            f.write(f"\n")
            
            # Print answers
            for mode in modes:
                run = q["runs"][mode]
                f.write(f"#### [{mode.upper()}]\n")
                quoted_gen = "\n".join([f"> {line}" for line in run["generation"].split("\n")])
                f.write(f"{quoted_gen}\n\n")
            
            f.write(f"---\n\n")
            
    print(f"\n[Success] Markdown report written to: {report_path}")
    return report_path

def mode_description(mode, profile=None):
    """Human-readable description for a run, optionally tagged with the embedder."""
    pname = f" [{profile}]" if profile else ""
    if mode == "baseline":
        return f"Baseline RAG{pname} (Chroma top {RERANK_TOP_K}, no rerank + LLM)"
    if mode == "reranked":
        return f"Two-stage RAG{pname} (Chroma top {RERANK_CANDIDATES} -> rerank to {RERANK_TOP_K} + LLM)"
    if mode == "expanded":
        return f"Full pipeline{pname} (rerank to {RERANK_TOP_K} -> page expansion +/-{CONTEXT_WINDOW} + LLM)"
    if mode == "direct_llm":
        return "Direct LLM (No context retrieval)"
    return ""

def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluation on Alexandria test queries.")
    parser.add_argument("--mode", type=str, default="baseline",
                        help="Comma-separated configurations to run: baseline, reranked, expanded, direct_llm, or 'all'")
    parser.add_argument("--embedder", type=str, default=DEFAULT_PROFILE,
                        help=f"Embedding profile(s) for retrieval modes: {', '.join(EMBEDDING_PROFILES)}, 'both', or 'all'")
    parser.add_argument("--queries", type=str, default="all",
                        help="Comma-separated query numbers (e.g., '1,2,5') or range ('1-5') or 'all'")
    parser.add_argument("--queries-file", type=str, default="./Docs/Test_Queries.txt",
                        help="Path to the queries text file")
    args = parser.parse_args()

    # Determine modes to run
    all_available_modes = ["baseline", "reranked", "expanded", "direct_llm"]
    if args.mode.lower() == "all":
        modes_to_run = all_available_modes
    else:
        modes_to_run = [m.strip() for m in args.mode.split(",") if m.strip() in all_available_modes]
        if not modes_to_run:
            print(f"Error: No valid modes specified in '{args.mode}'. Choose from: {all_available_modes}")
            sys.exit(1)

    # Determine embedding profiles to test
    if args.embedder.lower() in ("both", "all"):
        selected_embedders = list(EMBEDDING_PROFILES)
    else:
        selected_embedders = [e.strip() for e in args.embedder.split(",") if e.strip() in EMBEDDING_PROFILES]
        if not selected_embedders:
            print(f"Error: No valid embedders in '{args.embedder}'. Choose from: {list(EMBEDDING_PROFILES)}, 'both'")
            sys.exit(1)

    # Load each profile's vector DB (skip profiles whose DB hasn't been built yet)
    vdbs = {}
    for prof in selected_embedders:
        path = EMBEDDING_PROFILES[prof]["path"]
        if not os.path.exists(path):
            print(f"Warning: DB for profile '{prof}' not found at {path}. "
                  f"Build it with: python alexandria_vector_db.py --profile {prof}")
            continue
        vdbs[prof] = load_vector_db(prof)

    retrieval_modes = [m for m in modes_to_run if m != "direct_llm"]
    if retrieval_modes and not vdbs:
        print("Error: No embedding databases available for the requested retrieval modes.")
        sys.exit(1)

    # Build run specs: (label, mode, profile). direct_llm is embedder-independent.
    multi = len(vdbs) > 1
    run_specs = []
    descriptions = {}
    for mode in modes_to_run:
        if mode == "direct_llm":
            run_specs.append(("direct_llm", "direct_llm", None))
            descriptions["direct_llm"] = mode_description("direct_llm")
        else:
            for prof in vdbs:
                label = f"{mode}[{prof}]" if multi else mode
                run_specs.append((label, mode, prof))
                descriptions[label] = mode_description(mode, prof if multi else None)

    # Load and filter queries
    loaded_queries = load_queries(args.queries_file)
    if not loaded_queries:
        sys.exit(1)

    selected_queries = parse_queries_arg(args.queries, loaded_queries)

    print("=" * 60)
    print(f"ALEXANDRIA RAG BASELINE EVALUATION HARNESS")
    print(f"Loaded {len(loaded_queries)} queries from: {args.queries_file}")
    print(f"Running {len(selected_queries)} selected queries")
    print(f"Configurations to test: {', '.join(modes_to_run)}")
    print(f"Embedding profiles: {', '.join(vdbs) if vdbs else '(none / direct_llm only)'}")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    for q_id, q_text in selected_queries:
        print(f"\n[Query {q_id}] Running: '{q_text}'")
        query_result = {
            "id": q_id,
            "question": q_text,
            "runs": {}
        }

        for label, mode, prof in run_specs:
            print(f"  -> Executing config: {label}...", end="", flush=True)
            try:
                run_data = run_mode(mode, q_text, vdbs.get(prof) if prof else None)
                query_result["runs"][label] = run_data
                print(f" Done ({run_data['latency_s']:.2f}s)")
            except Exception as e:
                print(f" ERROR: {e}")
                query_result["runs"][label] = {
                    "generation": f"Error: Exception occurred during evaluation run: {e}",
                    "latency_s": 0.0,
                    "citations": [],
                    "num_sources": 0
                }

        results.append(query_result)

    # Write outputs
    os.makedirs(OUTPUT_DIR + MK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR + LOG_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR + LOG_DIR, f"results_baseline_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Success] Detailed JSON results saved to: {json_path}")
    
    run_labels = [label for label, _, _ in run_specs]
    report_path = write_markdown_report(results, run_labels, timestamp, descriptions)
    print("=" * 60)
    print("Evaluation completed successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()
