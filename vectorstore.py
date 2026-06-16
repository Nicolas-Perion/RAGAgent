import urllib.request
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from config import AWS_REGION

VECTORSTORE_PATH = "/tmp/vectorstore.json"
S3_URL = "https://xyz-corporation-893721683278-eu-north-1-an.s3.eu-north-1.amazonaws.com/vectorstore.json"

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

if not os.path.exists(VECTORSTORE_PATH):
    urllib.request.urlretrieve(S3_URL, VECTORSTORE_PATH)

vectorstore = InMemoryVectorStore.load(VECTORSTORE_PATH, embeddings)