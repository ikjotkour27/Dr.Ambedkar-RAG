import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------- CONFIG ----------------
CHUNKS_FILE = "prepared_chunks.json"
INDEX_FILE = "faiss.index"
METADATA_FILE = "faiss_metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"
# ----------------------------------------

def main():
    # Load chunks
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]

    metadata = [
        {
            "text": c["text"],
            "metadata": c["metadata"]
        }
        for c in chunks
    ]

    print(f"Loaded {len(texts)} chunks")

    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Create embeddings
    print("Creating embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")

    # Build FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print(f"FAISS index contains {index.ntotal} vectors")

    # Save index + metadata
    faiss.write_index(index, INDEX_FILE)

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Saved FAISS index and metadata")

if __name__ == "__main__":
    main()