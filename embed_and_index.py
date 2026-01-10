from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import json

client = QdrantClient(
    url="YOUR_QDRANT_URL",
    api_key="YOUR_QDRANT_API_KEY"
)

model = SentenceTransformer("all-MiniLM-L3-v2")

with open("prepared_chunks.json") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]
vectors = model.encode(texts).tolist()

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
