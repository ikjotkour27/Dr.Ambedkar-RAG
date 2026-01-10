from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

model = SentenceTransformer("all-MiniLM-L3-v2")

def retrieve(query):
    vec = model.encode(query).tolist()
    results = client.search(
        collection_name="ambedkar_rag",
        query_vector=vec,
        limit=3
    )
    return [r.payload for r in results]
