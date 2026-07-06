import operator
import time
from typing import Annotated, List, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from alexandria.config import (
    CHROMA_PATH,
    EMBEDDING_MODEL,
    RERANK_CANDIDATES,
    RERANK_TOP_K,
    ENABLE_CONTEXT_EXPANSION,
    CONTEXT_WINDOW,
    GROUNDING_MAX_RETRIES,
)
from alexandria.core.conversation_logger import ConversationLogger
from alexandria.core.structured import invoke_enum, strip_think
from alexandria.retrieval.reranker import CrossEncoderReranker
from alexandria.retrieval.context_expansion import expand_contiguous_pages

# --- 1. STATE DEFINITION ---
class AgentState(TypedDict):
    question: str
    classification: str
    # Annotated with operator.add allows nodes to append to this list
    documents: Annotated[List[str], operator.add]
    generation: str
    # Self-RAG grounding loop bookkeeping
    grounded: str            # last grounding verdict: "yes" / "no" / "caveat"
    grounding_attempts: int  # regenerations triggered so far (capped)

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
    # GBNF-constrained: the model physically cannot answer anything other than
    # {"classification": "manual_search"} or {"classification": "chitchat"}.
    classification = invoke_enum(
        llm,
        prompt,
        field="classification",
        values=("manual_search", "chitchat"),
        default="manual_search",  # safe choice for a knowledge-base assistant
    )
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

    system = (
        "Answer the user's question based ONLY on the context below. "
        "Each context block is tagged with its source document and page number. "
        "Cite the page number(s) you used in your answer. "
        "If the context does not contain the answer, say so explicitly.\n\n"
    )

    # Strict retry pass: the previous answer failed the grounding check.
    if state.get("grounding_attempts", 0) > 0:
        print("(strict regeneration after failed grounding check)")
        system = (
            "Your previous answer contained claims that are NOT supported by the "
            "context. Rewrite it now using ONLY facts explicitly present in the "
            "context below — do not add background knowledge, do not infer beyond "
            "what is written. If the context does not fully answer the question, "
            "say exactly which part is missing. "
            "Cite the page number(s) for every claim.\n\n"
        )

    context = "\n\n".join(documents)
    prompt = [
        SystemMessage(content=system + f"CONTEXT:\n{context}"),
        HumanMessage(content=state["question"])
    ]
    response = llm.invoke(prompt)
    return {"generation": response.content}

def grounding_node(state: AgentState):
    """Self-RAG check: is every claim in the answer supported by the context?

    Verdicts: 'yes' (accept), 'no' (regenerate, capped), 'caveat' (cap reached —
    accept but append an explicit warning so the user knows)."""
    print("--- CHECKING GROUNDING ---")
    documents = state["documents"]
    answer = strip_think(state["generation"])
    attempts = state.get("grounding_attempts", 0)

    # Nothing was retrieved: the generator already answered "not found" —
    # there are no claims to verify.
    if not documents:
        return {"grounded": "yes"}

    context = "\n\n".join(documents)
    prompt = [
        SystemMessage(content=(
            "You are a strict fact-checker for a technical-manual assistant. "
            "You are given CONTEXT excerpts from the manual and an ANSWER. "
            "Decide whether every factual claim in the ANSWER is supported by the "
            "CONTEXT. Judge only factual support, not style or completeness. "
            'Respond with ONLY a JSON object: {"verdict": "yes"} if fully '
            'supported, {"verdict": "no"} if any claim is unsupported.'
        )),
        HumanMessage(content=(
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {state['question']}\n\n"
            f"ANSWER:\n{answer}"
        )),
    ]
    # Default "yes": if the verdict infrastructure itself fails, degrade to the
    # non-agentic behavior (accept) rather than punishing a possibly fine answer.
    verdict = invoke_enum(
        llm, prompt, field="verdict", values=("yes", "no"), default="yes"
    )
    print(f"Grounding verdict: {verdict} (attempt {attempts})")

    if verdict == "yes":
        return {"grounded": "yes"}

    if attempts < GROUNDING_MAX_RETRIES:
        # Loop back to generate with the strict prompt.
        return {"grounded": "no", "grounding_attempts": attempts + 1}

    # Retry budget exhausted: accept, but say so honestly.
    caveat = (
        "\n\n[Warning: this answer did not pass the automatic grounding check "
        "against the retrieved manual pages. Verify the cited pages before "
        "relying on it.]"
    )
    return {"grounded": "caveat", "generation": state["generation"] + caveat}

def chitchat_node(state: AgentState):
    print("--- HANDLING CHITCHAT ---")
    response = llm.invoke([HumanMessage(content=state["question"])])
    return {"generation": response.content}

# --- 4. CONDITIONAL LOGIC (The Edges) ---
def decide_next_node(state: AgentState):
    if state["classification"] == "manual_search":
        return "retrieve"
    return "chitchat"

def decide_after_grounding(state: AgentState):
    # 'no' means: failed the check and retry budget remains -> regenerate.
    # 'yes' / 'caveat' both terminate.
    if state["grounded"] == "no":
        return "regenerate"
    return "accept"

# --- 5. GRAPH CONSTRUCTION ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("router", router_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generator_node)
workflow.add_node("check_grounding", grounding_node)
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

# Complete the paths. Generation is now followed by the Self-RAG grounding
# check, which can loop back to generate (strict retry) before terminating.
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "check_grounding")
workflow.add_conditional_edges(
    "check_grounding",
    decide_after_grounding,
    {
        "regenerate": "generate",
        "accept": END,
    }
)
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