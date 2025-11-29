from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

COLLECTION = "news_vectors"
SIZE = 768

client = QdrantClient("localhost", port=6333)

print("🧨 Deleting collection…")
client.delete_collection(COLLECTION)

print("✨ Recreating fresh collection…")
client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=SIZE, distance=Distance.COSINE)
)

print("🔥 Collection fresh as hell ✅")
print(client.get_collection(COLLECTION))
