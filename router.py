import json
import operator
from typing import Annotated, List, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from config import CHROMA_PATH, EMBEDDING_MODEL

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
RETRIEVAL_K = 5

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# --- 3. NODES ---

def router_node(state: AgentState):
    print("--- ROUTING QUESTION ---")
    question = state["question"]
    prompt = [
        SystemMessage(content="Classify the user question as 'manual_search' or 'chitchat'. Respond only in JSON: {'classification': 'value'}"),
        HumanMessage(content=question)
    ]
    response = llm.invoke(prompt)
    decision = json.loads(response.content)
    return {"classification": decision["classification"]}

def retrieve_node(state: AgentState):
    print("--- RETRIEVING FROM CHROMADB ---")
    results = vector_db.similarity_search(state["question"], k=RETRIEVAL_K)

    if not results:
        print("No relevant documents found.")
        return {"documents": []}

    retrieved_docs = []
    for doc in results:
        page = doc.metadata.get("source_page", "unknown")
        document_name = doc.metadata.get("document_name", "manual")
        retrieved_docs.append(f"[Source: {document_name}, Page {page}]\n{doc.page_content}")

    print(f"Retrieved {len(retrieved_docs)} chunks.")
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