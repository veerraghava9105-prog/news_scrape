# scripts/bootstrap_qdrant_from_npz.py

import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parents[1]  # .../project
DATA_DIR = BASE_DIR / "data"
EMB_PATH = DATA_DIR / "embeddings.npz"
INDEX_PATH = DATA_DIR / "embeddings_index.json"

COLLECTION = "news"

def main():
    if not EMB_PATH.exists():
        raise SystemExit(f"Embeddings file not found: {EMB_PATH}")
    if not INDEX_PATH.exists():
        raise SystemExit(f"Index file not found: {INDEX_PATH}")

    print(f"📂 Loading embeddings from {EMB_PATH}")
    data = np.load(EMB_PATH)
    # Adjust this if your key is different
    embeddings = data["embeddings"] if "embeddings" in data else list(data.values())[0]
    dim = embeddings.shape[1]
    print(f"✅ Embeddings shape: {embeddings.shape}")

    print(f"📂 Loading index from {INDEX_PATH}")
    with INDEX_PATH.open("r", encoding="utf-8") as f:
        index = json.load(f)

    if len(index) != embeddings.shape[0]:
        raise SystemExit("❌ index length != embeddings rows — mismatch, check your files")

    print("🔌 Connecting to Qdrant at localhost:6333 ...")
    client = QdrantClient(url="http://localhost:6333")

    # Create collection if not exists
    if not client.collection_exists(COLLECTION):
        print(f"🆕 Creating collection '{COLLECTION}' with dim={dim}")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE
            ),
        )
    else:
        print(f"ℹ️ Collection '{COLLECTION}' already exists, will upsert into it")

    # Build points
    points = []
    for emb_row, meta in zip(embeddings, index):
        article_id = meta.get("id") or meta.get("url")
        if not article_id:
            continue

        payload = {
            "article_id": article_id,
            "title": meta.get("title"),
            "url": meta.get("url"),
            "source": meta.get("source"),
        }

        points.append(
            PointStruct(
                id=article_id,  # use article ID as point ID
                vector=emb_row.astype(float).tolist(),
                payload=payload,
            )
        )

    print(f"🚀 Upserting {len(points)} points into Qdrant...")
    client.upsert(
        collection_name=COLLECTION,
        points=points,
    )
    print("✅ Done seeding Qdrant. Vector DB is now in sync with existing npz/index.")

if __name__ == "__main__":
    main()
