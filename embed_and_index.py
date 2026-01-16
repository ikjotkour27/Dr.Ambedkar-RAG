from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60
)

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("prepared_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]
vectors = model.encode(texts, show_progress_bar=True).tolist()

points = []
for i, c in enumerate(chunks):
    points.append({
        "id": i,
        "vector": vectors[i],
        "payload": c
    })

client.upsert(
    collection_name="ambedkar_rag",
    points=points
)

print("Vectors uploaded to Qdrant")
