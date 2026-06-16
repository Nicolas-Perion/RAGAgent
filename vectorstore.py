from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DOCUMENT_PATH, CHUNK_SIZE, CHUNK_OVERLAP

load_dotenv()

loader = Docx2txtLoader(DOCUMENT_PATH)

docs = loader.load()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = InMemoryVectorStore(embeddings)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)

splits = text_splitter.split_documents(docs)
_ = vectorstore.add_documents(documents=splits)

vectorstore.dump("vectorstore.json")