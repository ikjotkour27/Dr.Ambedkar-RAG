import json
import faiss
import numpy as np
import subprocess
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
INDEX_FILE = "faiss.index"
METADATA_FILE = "faiss_metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3
OLLAMA_MODEL = "mistral"
# ----------------------------------------

def load_index():
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

def embed_query(query, model):
    return model.encode([query], convert_to_numpy=True)

def retrieve(query, model, index, metadata, top_k):
    query_vec = embed_query(query, model)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for idx in indices[0]:
        results.append(metadata[idx])

    return results

def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(
        f"- {chunk['source']}:\n{chunk['text']}"
        for chunk in retrieved_chunks
    )

    prompt = f"""
You are an assistant answering ONLY from the provided context.
Do NOT use outside knowledge.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt.strip()

def call_llm(prompt):
    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()

def main():
    print("Loading model and index...")
    model = SentenceTransformer(MODEL_NAME)
    index, metadata = load_index()

    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() == "exit":
            break

        retrieved = retrieve(query, model, index, metadata, TOP_K)
        prompt = build_prompt(query, retrieved)

        print("\nThinking...\n")
        answer = call_llm(prompt)

        print("ANSWER:\n")
        print(answer)

if __name__ == "__main__":
    main()