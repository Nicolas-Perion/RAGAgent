import boto3
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

VECTORSTORE_PATH = "/tmp/vectorstore.json"
S3_BUCKET = "xyz-corporation-893721683278-eu-north-1-an"
S3_KEY = "vectorstore.json"

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

if not os.path.exists(VECTORSTORE_PATH):
    s3 = boto3.client("s3", region_name="eu-north-1")
    s3.download_file(S3_BUCKET, S3_KEY, VECTORSTORE_PATH)
    
vectorstore = InMemoryVectorStore.load(VECTORSTORE_PATH, embeddings)