# ingest.py
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def load_documents():
    docs = []
    for file in os.listdir(DATA_PATH):
        path = os.path.join(DATA_PATH, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif file.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
    return docs

def create_vectorstore():
    docs = load_documents()

    # Découpage en chunks (500 caractères environ)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Embeddings avec SentenceTransformers
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding=model)

    vectorstore.save_local(DB_FAISS_PATH)
    print("✅ Base de données vectorielle créée !")

if __name__ == "__main__":
    create_vectorstore()
