import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer


ARTICLES_PATH = "../data/articles.jsonl"
EMBEDDINGS_PATH = "../data/embeddings.jsonl"


def load_articles(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    print("Loading model... chill for 3-5 sec first time")
    start = time.time()
    model = SentenceTransformer("all-MiniLM-L6-v2")   # loads ONCE here
    print("Model loaded in", time.time() - start, "seconds")

    articles = list(load_articles(ARTICLES_PATH))
    texts = [
        (a["id"], a["title"] + "\n\n" + a["text"][:1000])
        for a in articles
    ]

    print(f"Embedding {len(texts)} articles...")

    start = time.time()
    emb_vectors = model.encode(
        [t[1] for t in texts],
        batch_size=32,                       # SPEED BOOST
        normalize_embeddings=True,
        show_progress_bar=True
    )
    print("Embedding complete in", time.time() - start, "seconds")

    with open(EMBEDDINGS_PATH, "w", encoding="utf-8") as fout:
        for (id_, _), emb in zip(texts, emb_vectors):
            fout.write(json.dumps({
                "id": id_,
                "embedding": emb.tolist()
            }) + "\n")

if __name__ == "__main__":
    main()
