from langchain.chat_models import init_chat_model
from tools import retrieve_context
from pydantic import BaseModel, Field
from typing import Literal

# Base model for all the agents
base_model = init_chat_model(
    model="gemini-2.5-flash-lite",
    model_provider="google_genai"
)

model_with_tools = base_model.bind_tools([retrieve_context])

# Document grader agent
class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: Literal["yes", "no"] = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

model_DG = base_model.with_structured_output(GradeDocuments)