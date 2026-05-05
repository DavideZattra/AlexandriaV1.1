import json
import operator
from typing import Annotated, List, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

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
    # Logic to query your vector store (as we discussed previously)
    # For now, we simulate finding 2 chunks
    retrieved_docs = ["TCPOS Manual Section 1.2: How to open a shift...", "TCPOS Manual Section 4: Printer config..."]
    return {"documents": retrieved_docs}

def generator_node(state: AgentState):
    print("--- GENERATING ANSWER ---")
    context = "\n".join(state["documents"])
    prompt = [
        SystemMessage(content=f"Answer based ONLY on this context: {context}"),
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