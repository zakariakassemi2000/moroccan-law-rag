from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

DB_FAISS_PATH = "vectorstore/db_faiss"

# 1️⃣ Charger embeddings + vectorstore (compatible FAISS)
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(DB_FAISS_PATH, embedder, allow_dangerous_deserialization=True)

# 2️⃣ Charger un modèle léger pour génération de texte (Flan-T5-small)
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
qa_model = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# 3️⃣ Initialiser Flask
app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    # 4️⃣ Récupérer la question depuis le JSON
    query = request.json.get("question")
    if not isinstance(query, str):
        return {"error": "Question must be a string"}, 400
    
    # 5️⃣ Recherche des passages les plus proches
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])

    # 6️⃣ Construire le prompt pour le modèle
    prompt = (
        f"Tu es un assistant juridique spécialisé dans le droit marocain.\n"
        f"Réponds en francais de manière claire et concise.\n"
        f"Formate les références aux articles de loi avec 'Article X'.\n\n"
        f"Contexte :\n{context}\n\n"
        f"Question : {query}\nRéponse :"
    )

    # 7️⃣ Génération de la réponse (réponse courte)
    result = qa_model(prompt, max_length=150, do_sample=False)
    answer = result[0]["generated_text"]

    # 8️⃣ Retourner la réponse et les sources
    return jsonify({
        "answer": answer.strip(),
        "sources": [
            {
                "source": d.metadata.get("source", "inconnu"),
                "page": d.metadata.get("page_label", "?")
            } for d in docs
        ]
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
