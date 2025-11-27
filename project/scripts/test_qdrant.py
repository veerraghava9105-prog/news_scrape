from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import numpy as np

client = QdrantClient(host="localhost", port=6333)

COLLECTION = "test_news"
VECTOR_DIM = 768

# Only create if not already there
existing = [c.name for c in client.get_collections().collections]
if COLLECTION not in existing:
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    print(f"✅ Created collection: {COLLECTION}")
else:
    print(f"⚡ Skipped creation, using existing: {COLLECTION}")

def main():
    print("📌 Upserting 3 dummy vectors...")
    dummy_vecs = np.random.random((3, VECTOR_DIM))

    points = [
        PointStruct(id=i, vector=vec.tolist(), payload={"dummy": True})
        for i, vec in enumerate(dummy_vecs)
    ]

    client.upsert(collection_name=COLLECTION, points=points)
    print("✅ Upserted dummy vectors")

    print("🔎 Running cosine similarity search...")

    query_vec = np.random.random(VECTOR_DIM).tolist()

    # 💎 THIS IS THE REAL FIX — similarity search
    res = client.search_points(
        collection_name=COLLECTION,
        query_vector=query_vec,
        limit=5,
        with_payload=True,
    )

    print("💎✅ Results:")
    for r in res:
        print(f" → id: {r.id}, score: {r.score:.4f}, payload: {r.payload}")

if __name__ == "__main__":
    main()
