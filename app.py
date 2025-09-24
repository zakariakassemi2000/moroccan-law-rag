# app.py
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import pipeline

DB_FAISS_PATH = "vectorstore/db_faiss"

# Charger embeddings + vectorstore
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(DB_FAISS_PATH, embedder, allow_dangerous_deserialization=True)

# Charger un modèle de langage (Mistral)
qa_model = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct", device=-1)

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    query = request.json.get("question")
    docs = vectorstore.similarity_search(query, k=3)  # Top 3 passages les plus proches
    context = "\n".join([d.page_content for d in docs])

    prompt = f"Tu es un assistant juridique spécialisé dans le droit marocain.\n\nContexte:\n{context}\n\nQuestion: {query}\nRéponse:"
    
    result = qa_model(prompt, max_new_tokens=300, temperature=0.2)
    answer = result[0]["generated_text"]

    return jsonify({"answer": answer, "sources": [d.metadata for d in docs]})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
