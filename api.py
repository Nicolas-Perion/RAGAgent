import boto3
import json
import os
from config import AWS_REGION
from fastapi.staticfiles import StaticFiles

client = boto3.client("secretsmanager", region_name= AWS_REGION)
secret = client.get_secret_value(SecretId="RAG-API-secret")
secrets = json.loads(secret["SecretString"])

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = secrets["LANGCHAIN_PROJECT"]
os.environ["GOOGLE_API_KEY"] = secrets["GOOGLE_API_KEY"]
os.environ["S3_BUCKET"] = secrets["S3_BUCKET"]

from langsmith import Client
Client()

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

app.mount("/", StaticFiles(directory="static", html=True), name="static")