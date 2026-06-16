from state import RAGState
from langgraph.graph import END, START, StateGraph
from nodes import RAG_or_memory, rewrite_question, generate_answer_with_context, grade_documents
from langgraph.prebuilt import ToolNode
from tools import retrieve_context
from agents import model_DG
from typing import Literal
from PIL import Image as PILImage
import io

def route(state: RAGState) -> Literal["retriever", "end"]:
    """
    Depending on the last message, route the graph either to the 'retriever' node
    or to the END node.

    Args:
    state (RAGState): The current graph state regarding messages.
    
    Returns:
        Literal["retriever", "end"]: The name of the next node.
    """
    last_message = state["messages"][-1]
    return "retriever" if getattr(last_message, "tool_calls", None) else "end"

pipeline = StateGraph(RAGState)

pipeline.add_node(RAG_or_memory)
pipeline.add_node("retriever", ToolNode([retrieve_context]))
pipeline.add_node(rewrite_question)
pipeline.add_node(generate_answer_with_context)

pipeline.add_edge(START,"RAG_or_memory")

pipeline.add_conditional_edges(
    "RAG_or_memory",
    route,
    {
        "retriever": "retriever",
        "end": END
    }
)

pipeline.add_conditional_edges(
    "retriever",
    grade_documents
)

pipeline.add_edge("rewrite_question", "RAG_or_memory")
pipeline.add_edge("generate_answer_with_context", END)

graph = pipeline.compile()

# Uncomment if you want to see the graph schema.
# png_data = graph.get_graph().draw_mermaid_png()
# image = PILImage.open(io.BytesIO(png_data))
# image.show()