from pathlib import Path
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"

COLLECTION_NAME = "news_vectors"
MODEL_NAME = "BAAI/bge-base-en-v1.5"

def main():
    client = QdrantClient("localhost", port=6333)
    model = SentenceTransformer(MODEL_NAME)

    while True:
        query = input("\n🔍 Enter query (or 'q' to quit): ").strip()
        if not query or query.lower() == "q":
            break

        q_vec = model.encode([query], normalize_embeddings=True)[0]

        res = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=q_vec.tolist(),
            limit=5,
            with_payload=True
        )

        print("\n🔥 Top results:")
        for hit in res:
            payload = hit.payload or {}
            print(f"  • score: {hit.score:.3f}")
            print(f"    ➤ {payload.get('title','<no title>')}")
            print(f"    🌐 {payload.get('url','<no url>')}")
            print()

if __name__ == "__main__":
    main()
