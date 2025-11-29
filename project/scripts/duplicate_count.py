from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)
pts = client.scroll(collection_name="news_vectors", limit=10000, with_payload=True)[0]

stored_ids = [pt.payload.get("id") for pt in pts]
print("Total points in Qdrant:", len(stored_ids))
print("Unique IDs in Qdrant:", len(set(stored_ids)))
