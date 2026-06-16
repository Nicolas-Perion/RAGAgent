from langchain.tools import tool
from vectorstore import vectorstore
from config import RETRIEVER_K

@tool
def retrieve_context(query: str) -> str:
    """
    Retrieve information to help answer a query.

    Args:
        query (str): The query provided by the user.

    Returns:
        str: A chain of strings containing the metadata and content of each document used to answer the query.
    """
    retrieved_docs = vectorstore.similarity_search(query, k=RETRIEVER_K)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized