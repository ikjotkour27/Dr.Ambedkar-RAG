import json
import faiss
import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
INDEX_FILE = "faiss.index"
METADATA_FILE = "faiss_metadata.json"
EMBED_MODEL = "all-MiniLM-L3-v2"   # same model used to build FAISS
GEMINI_MODEL = "models/gemini-flash-latest"
TOP_K = 3
# ----------------------------------------

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel(GEMINI_MODEL)

# Load embedding model ONCE
embed_model = SentenceTransformer(EMBED_MODEL)

# Load FAISS and metadata ONCE
print("Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)

print("Loading metadata...")
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)


def retrieve(query, k):
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)

    results = []
    for idx in indices[0]:
        results.append(metadata[idx])

    return results


def build_prompt(query, chunks):
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
    response = gemini.generate_content(prompt)
    return response.text


def answer_question(query: str) -> str:
    retrieved = retrieve(query, TOP_K)
    prompt = build_prompt(query, retrieved)
    return generate_answer(prompt)
