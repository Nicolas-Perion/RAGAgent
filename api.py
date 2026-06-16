from fastapi import FastAPI
from graph import graph

app = FastAPI()

@app.get("/query")
def query(question: str) -> dict:
    """
    Get an answer from the graph architecture through the API.

    Args:
        question (str): The user's question.

    Returns:
        dict: The answer.
    """
    graph_input = {"messages": [{"role": "user", "content": question}]}
    result = graph.invoke(graph_input)
    return {"answer": result["messages"][-1].content}

@app.get("/health")
def health() -> dict:
    """
    Function just to indicate that the server is running.

    Returns:
        dict: The status.
    """
    return {"status": "ok"}