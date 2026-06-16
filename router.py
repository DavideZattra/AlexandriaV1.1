import json
import operator
import re
import time
from typing import Annotated, List, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from config import (
    CHROMA_PATH,
    EMBEDDING_MODEL,
    RERANK_CANDIDATES,
    RERANK_TOP_K,
    ENABLE_CONTEXT_EXPANSION,
    CONTEXT_WINDOW,
)
from conversation_logger import ConversationLogger
from reranker import CrossEncoderReranker
from context_expansion import expand_contiguous_pages

# --- 1. STATE DEFINITION ---
class AgentState(TypedDict):
    question: str
    classification: str
    # Annotated with operator.add allows nodes to append to this list
    documents: Annotated[List[str], operator.add]
    generation: str

# --- 2. LLM SETUP (llama.cpp) ---
llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
    temperature=0
)

# --- 2b. VECTOR STORE SETUP (ChromaDB) ---
# RETRIEVAL_K is the Stage-1 candidate pool size. The cross-encoder reranker
# (Stage 2) then narrows it down to RERANK_TOP_K before generation.
RETRIEVAL_K = RERANK_CANDIDATES

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# --- 2c. CROSS-ENCODER RERANKER SETUP (Stage 2) ---
reranker = CrossEncoderReranker()

# --- 3. NODES ---

def _parse_classification(raw: str) -> str:
    """Robustly extract the classification from a noisy LLM response.

    Local models often wrap JSON in markdown fences or add stray text, so we
    cannot rely on json.loads alone. We try strict JSON first, then fall back
    to keyword detection, defaulting to 'manual_search' (the safe choice for a
    knowledge-base assistant)."""
    text = raw.strip()

    # 1. Try to find a JSON object anywhere in the response
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            decision = json.loads(match.group(0))
            value = str(decision.get("classification", "")).lower()
            if "chitchat" in value:
                return "chitchat"
            if "manual" in value:
                return "manual_search"
        except json.JSONDecodeError:
            pass

    # 2. Fall back to keyword detection on the raw text
    lowered = text.lower()
    if "chitchat" in lowered:
        return "chitchat"

    # 3. Default to manual_search (safer for a manual-querying assistant)
    return "manual_search"


def router_node(state: AgentState):
    print("--- ROUTING QUESTION ---")
    question = state["question"]
    prompt = [
        SystemMessage(content=(
            "You are a query classifier. Classify the user question into exactly one "
            "of two categories:\n"
            "- 'manual_search': the user is asking about the product/manual content.\n"
            "- 'chitchat': greetings, small talk, or questions unrelated to the manual.\n"
            'Respond with ONLY a JSON object, no other text: {"classification": "manual_search"}'
        )),
        HumanMessage(content=question)
    ]
    response = llm.invoke(prompt)
    classification = _parse_classification(response.content)
    print(f"Classified as: {classification}")
    return {"classification": classification}

def retrieve_node(state: AgentState):
    print("--- RETRIEVING FROM CHROMADB ---")
    question = state["question"]

    # Stage 1: bi-encoder retrieval — cast a wide net of candidates.
    candidates = vector_db.similarity_search(question, k=RERANK_CANDIDATES)

    if not candidates:
        print("No relevant documents found.")
        return {"documents": []}

    # Stage 2: cross-encoder reranking — re-score and keep the most relevant.
    print(f"--- RERANKING {len(candidates)} CANDIDATES (keep top {RERANK_TOP_K}) ---")
    ranked = reranker.rerank(question, candidates, top_k=RERANK_TOP_K)

    # Stage 3 (optional): contiguous page context expansion.
    if ENABLE_CONTEXT_EXPANSION:
        print(f"--- EXPANDING CONTEXT (+/- {CONTEXT_WINDOW} page(s)) ---")
        retrieved_docs = expand_contiguous_pages(vector_db, ranked, window=CONTEXT_WINDOW)
    else:
        retrieved_docs = []
        for doc, score in ranked:
            page = doc.metadata.get("source_page", "unknown")
            document_name = doc.metadata.get("document_name", "manual")
            retrieved_docs.append(f"[Source: {document_name}, Page {page}]\n{doc.page_content}")

    print(f"Retrieved {len(retrieved_docs)} context blocks.")
    return {"documents": retrieved_docs}

def generator_node(state: AgentState):
    print("--- GENERATING ANSWER ---")
    documents = state["documents"]

    if not documents:
        return {"generation": "I could not find any relevant information in the manual to answer this question."}

    context = "\n\n".join(documents)
    prompt = [
        SystemMessage(content=(
            "Answer the user's question based ONLY on the context below. "
            "Each context block is tagged with its source document and page number. "
            "Cite the page number(s) you used in your answer. "
            "If the context does not contain the answer, say so explicitly.\n\n"
            f"CONTEXT:\n{context}"
        )),
        HumanMessage(content=state["question"])
    ]
    response = llm.invoke(prompt)
    return {"generation": response.content}

def chitchat_node(state: AgentState):
    print("--- HANDLING CHITCHAT ---")
    response = llm.invoke([HumanMessage(content=state["question"])])
    return {"generation": response.content}

# --- 4. CONDITIONAL LOGIC (The Edge) ---
def decide_next_node(state: AgentState):
    if state["classification"] == "manual_search":
        return "retrieve"
    return "chitchat"

# --- 5. GRAPH CONSTRUCTION ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("router", router_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generator_node)
workflow.add_node("chitchat", chitchat_node)

# Set Entry Point
workflow.set_entry_point("router")

# Add Conditional Edges
workflow.add_conditional_edges(
    "router",
    decide_next_node,
    {
        "retrieve": "retrieve",
        "chitchat": "chitchat"
    }
)

# Complete the paths
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
workflow.add_edge("chitchat", END)

# Compile the Graph
app = workflow.compile()


# --- 6. INTERACTIVE CHAT LOOP ---
def chat():
    logger = ConversationLogger()
    print("=== ALEXANDRIA RAG ===")
    print("Ask a question about the manual. Type 'exit' or 'quit' to leave.")
    print(f"Logging this session to: {logger.path}\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        start = time.perf_counter()
        try:
            result = app.invoke({"question": question})
            latency = time.perf_counter() - start
            print(f"\nAlexandria: {result['generation']}\n")
            logger.log_turn(question, result=result, latency_s=latency)
        except Exception as e:
            latency = time.perf_counter() - start
            print(f"\nERROR: {e}\n")
            logger.log_turn(question, latency_s=latency, error=str(e))


if __name__ == "__main__":
    chat()