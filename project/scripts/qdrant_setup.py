# scripts/qdrant_setup.py

import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Config
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "news_articles"
VECTOR_SIZE = 768  # BAAI/bge-base-en-v1.5

def main():
    client = QdrantClient(url=QDRANT_URL)

    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        print(f"✅ Collection '{COLLECTION_NAME}' already exists.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )
    print(f"🎯 Created collection '{COLLECTION_NAME}' with dim={VECTOR_SIZE}, metric=cosine")

if __name__ == "__main__":
    main()
