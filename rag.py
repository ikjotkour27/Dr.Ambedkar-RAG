# rag.py
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

model = SentenceTransformer("all-MiniLM-L3-v2")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-pro")


def retrieve(query, top_k=3):
    vec = model.encode(query).tolist()
    results = client.search(
        collection_name="ambedkar_rag",
        query_vector=vec,
        limit=top_k
    )
    return [r.payload for r in results]


def answer_question(question):
    contexts = retrieve(question)

    context_text = "\n\n".join(
        f"Source: {c['metadata']['source']}\nText: {c['text']}"
        for c in contexts
    )

    prompt = f"""
You are a scholarly assistant answering questions using Dr. B. R. Ambedkar's writings.

Context:
{context_text}

Question:
{question}

Answer in a clear, concise and academic tone. If the answer is not found in the context, say so.
"""

    response = gemini.generate_content(prompt)
    return response.text.strip()
