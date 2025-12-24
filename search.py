import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -------- CONFIG --------
INDEX_FILE = "faiss.index"
METADATA_FILE = "faiss_metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3
# ------------------------

def load_index():
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

def search(query, index, metadata, model, top_k=3):
    # Embed query
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Search FAISS
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        chunk = metadata[idx]
        results.append({
            "rank": i + 1,
            "distance": float(distances[0][i]),
            "text": chunk["text"],
            "metadata": chunk["metadata"]
        })

    return results

if __name__ == "__main__":
    print("Loading model and index...")
    model = SentenceTransformer(MODEL_NAME)
    index, metadata = load_index()

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        results = search(query, index, metadata, model, TOP_K)

        print("\n--- Top Results ---")
        for r in results:
            print(f"\nRank {r['rank']}")
            print(f"Source: {r['metadata']['source']}")
            print(f"Category: {r['metadata']['category']}")
            print(f"Text:\n{r['text']}")