from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    rewrite_count: int