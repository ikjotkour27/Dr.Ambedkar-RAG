from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(
    url="YOUR_QDRANT_URL",
    api_key="YOUR_QDRANT_API_KEY"
)

client.recreate_collection(
    collection_name="ambedkar_rag",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

print("Collection created")
