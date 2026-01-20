from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import os

print("RAG loaded")
load_dotenv()

# Qdrant
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60
)

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")



def retrieve(query, top_k=3):
    vector = embedder.encode(query).tolist()
    results = qdrant.query_points(
        collection_name="ambedkar_rag",
        prefetch=[],
        query=vector,
        limit=top_k,
        with_payload=True
    )
    return [p.payload for p in results.points]


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

Answer in a clear, concise and academic tone. 
If the answer is not found in the context, say so clearly.
"""

   response = model.generate_content(prompt)


    return response.text.strip()
