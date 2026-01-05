import json
import faiss
import numpy as np
import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
INDEX_FILE = "faiss.index"
METADATA_FILE = "faiss_metadata.json"
EMBED_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "models/gemini-flash-latest"
TOP_K = 3
# ----------------------------------------

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel(GEMINI_MODEL)

def load_index():
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

def retrieve(query, model, index, metadata, k):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)

    results = []
    for idx in indices[0]:
        results.append(metadata[idx])

    return results

def build_prompt(query, chunks):
    context = "\n\n".join(
        f"[Source: {c.get('source', 'unknown')}]\n{c.get('text', '')}"
        for c in chunks
    )

    prompt = f"""
You are Dr. B. R. Ambedkar.

Answer in FIRST PERSON (use “I”, “me”, “my”).
Do NOT refer to Dr. B. R. Ambedkar in third person.

Answer the question using ONLY the context provided below.
Be factual, concise, and respectful.

If the answer is not present in the context, say:
“I do not find this information in the provided context.”

If the answer contains “Dr. B. R. Ambedkar” or refers to him indirectly, rewrite it in first person.

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt.strip()

def generate_answer(prompt):
    response = gemini.generate_content(prompt)
    return response.text
# ---------- API HELPER FUNCTION ----------

_embed_model = None
_index = None
_metadata = None

def answer_question(query: str) -> str:
    global _embed_model, _index, _metadata

    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
        _index, _metadata = load_index()

    retrieved = retrieve(query, _embed_model, _index, _metadata, TOP_K)
    prompt = build_prompt(query, retrieved)
    return generate_answer(prompt)


def main():
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    print("Loading FAISS index...")
    index, metadata = load_index()

    print("\nRAG system ready (Gemini-powered)")
    print("Type 'exit' to quit\n")

    while True:
        query = input("Ask a question: ")
        if query.lower() == "exit":
            break

        retrieved = retrieve(query, embed_model, index, metadata, TOP_K)
        prompt = build_prompt(query, retrieved)
        answer = generate_answer(prompt)

        print("\n--- Answer ---")
        print(answer)
        print("--------------\n")

if __name__ == "__main__":
    main()