from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from modules.rag_pipeline import RAGPipeline
from modules.agent import GeneralAgent
from modules.general_llm import get_general_answer
from utils.logger import get_logger

logger = get_logger(__name__)

class AgentState(TypedDict):
    query: str
    response: str
    route: str

def router(state: AgentState):
    """Router node to decide where to go."""
    query = state["query"].lower()
    
    if any(word in query for word in ["calculate", "math", "search", "who is"]):
        return "agent"
    elif any(word in query for word in ["document", "pdf", "file", "based on context"]):
        return "rag"
    else:
        return "general"

def call_rag(state: AgentState):
    logger.info("Routing to RAG Pipeline")
    rag = RAGPipeline()
    res = rag.answer_question(state["query"])
    # If RAG returns source documents, format it
    if isinstance(res, dict) and "result" in res:
        return {"response": res["result"]}
    return {"response": str(res)}

def call_agent(state: AgentState):
    logger.info("Routing to Agent")
    agent = GeneralAgent()
    res = agent.run(state["query"])
    return {"response": res}

def call_general(state: AgentState):
    logger.info("Routing to General LLM")
    res = get_general_answer(state["query"])
    return {"response": res.content if hasattr(res, 'content') else str(res)}

def create_workflow():
    workflow = StateGraph(AgentState)
    
    # Define Nodes
    workflow.add_node("rag", call_rag)
    workflow.add_node("agent", call_agent)
    workflow.add_node("general", call_general)
    
    # Define Entry Point with Conditional Routing
    workflow.set_conditional_entry_point(
        router,
        {
            "rag": "rag",
            "agent": "agent",
            "general": "general"
        }
    )
    
    # All nodes go to END
    workflow.add_edge("rag", END)
    workflow.add_edge("agent", END)
    workflow.add_edge("general", END)
    
    return workflow.compile()
