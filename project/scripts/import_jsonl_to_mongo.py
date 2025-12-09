import os
import json
from pymongo import MongoClient, errors
from tqdm import tqdm
from datetime import datetime

# ---------------- CONFIG ----------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT = os.path.join(DATA_DIR, "articles.jsonl")

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "news_engine"        # <---- FINAL DB NAME
COLLECTION_NAME = "articles"   # <---- SAME COLLECTION

# ----------------------------------------

def normalize_old_doc(d):
    """Ensures old JSONL docs match the new Mongo schema."""
    return {
        "url": d.get("url"),
        "title": d.get("title") or "",
        "text": d.get("text") or "",
        "word_count": d.get("word_count") or len((d.get("text") or "").split()),

        "feed": d.get("feed"),
        "published": d.get("published"),

        "scraped_at": d.get("scraped_at") or datetime.utcnow(),
        "domain": d.get("domain") or "",

        # If old doc had no images → assign empty list
        "images": d.get("images", []),

        # Placeholder for future metadata
        "meta": d.get("meta", {})
    }


def main():
    print("DEBUG INPUT PATH =", INPUT)
    print("EXISTS =", os.path.exists(INPUT))

    if not os.path.exists(INPUT):
        print("\n❌ ERROR: articles.jsonl NOT FOUND\n")
        return

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    col = db[COLLECTION_NAME]

    # Create unique index on url
    try:
        col.create_index("url", unique=True)
    except Exception:
        pass

    print("\nStarting migration…\n")

    batch = []
    BATCH_SIZE = 500
    inserted = 0
    skipped = 0

    with open(INPUT, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Importing"):
            line = line.strip()
            if not line:
                continue

            try:
                doc = json.loads(line)
            except Exception:
                skipped += 1
                continue

            if "url" not in doc:
                skipped += 1
                continue

            # normalize to match new schema
            doc = normalize_old_doc(doc)

            try:
                col.insert_one(doc)
                inserted += 1
            except errors.DuplicateKeyError:
                skipped += 1
                continue
            except Exception:
                skipped += 1
                continue

    print(f"\n✨ Migration Completed")
    print(f"   ➤ Inserted: {inserted}")
    print(f"   ➤ Skipped:  {skipped}\n")


if __name__ == "__main__":
    main()
