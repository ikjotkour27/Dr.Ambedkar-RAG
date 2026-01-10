import json
import faiss
import numpy as np
import os
import google.generativeai as genai

# ---------------- CONFIG ----------------
INDEX_FILE = "faiss.index"
METADATA_FILE = "faiss_metadata.json"
GEMINI_MODEL = "models/gemini-flash-latest"
EMBED_MODEL = "models/embedding-001"
TOP_K = 3
# ----------------------------------------

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel(GEMINI_MODEL)

_index = None
_metadata = None


def load_index():
    """
    Load FAISS index and metadata only once.
    """
    global _index, _metadata
    if _index is None:
        print("Loading FAISS index...")
        _index = faiss.read_index(INDEX_FILE)

        print("Loading metadata...")
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            _metadata = json.load(f)

    return _index, _metadata


def embed_query(query: str):
    """
    Create embedding using Gemini embedding API.
    This avoids loading any heavy ML model locally.
    """
    embedding = genai.embed_content(
        model=EMBED_MODEL,
        content=query
    )
    return np.array([embedding["embedding"]], dtype="float32")


def retrieve(query, index, metadata, k):
    """
    Retrieve top-k relevant chunks from FAISS.
    """
    query_vec = embed_query(query)
    distances, indices = index.search(query_vec, k)

    results = []
    for idx in indices[0]:
        results.append(metadata[idx])

    return results


def build_prompt(query, chunks):
    """
    Build the RAG prompt in first-person style as Dr. B. R. Ambedkar.
    """
    context = "\n\n".join(
        f"[Source: {c.get('metadata', {}).get('source', 'unknown')}]\n{c.get('text', '')}"
        for c in chunks
    )

    prompt = f"""
You are Dr. B. R. Ambedkar.

Answer in FIRST PERSON (use “I”, “me”, “my”).
Do NOT refer to Dr. B. R. Ambedkar in third person.

Answer the question using ONLY the context provided below.
Be factual, concise, and respectful.

If the answer is not present in the context, say:
"I do not find this information in the provided context."

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt.strip()


def generate_answer(prompt):
    """
    Generate final answer using Gemini LLM.
    """
    response = gemini.generate_content(prompt)
    return response.text


def answer_question(query: str) -> str:
    """
    Main API helper called by FastAPI.
    """
    index, metadata = load_index()
    retrieved = retrieve(query, index, metadata, TOP_K)
    prompt = build_prompt(query, retrieved)
    return generate_answer(prompt)


# ---------------- LOCAL TEST MODE ----------------
if __name__ == "__main__":
    print("RAG system ready (Gemini-powered, memory-safe)")
    print("Type 'exit' to quit\n")

    while True:
        query = input("Ask a question: ")
        if query.lower() == "exit":
            break

        answer = answer_question(query)
        print("\n--- Answer ---")
        print(answer)
        print("--------------\n")
