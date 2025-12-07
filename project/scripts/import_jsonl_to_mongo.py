# import_jsonl_to_mongo.py
import json
import os
import sys
from pymongo import MongoClient, errors
from urllib.parse import urlparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(PROJECT_ROOT, "data")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = mongo.newsdb
articles_col = db.articles
failed_col = db.failed_articles

# make sure unique index exists
articles_col.create_index("url", unique=True)

def import_file(path):
    saved = 0
    skipped = 0
    total = 0

    if not os.path.exists(path):
        print("file not found:", path)
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                obj = json.loads(line)
            except:
                skipped += 1
                continue
            url = obj.get("url") or obj.get("link")
            if not url:
                skipped += 1
                continue

            # normalize minimal fields
            doc = {
                "url": url,
                "title": obj.get("title"),
                "text": obj.get("text"),
                "images": obj.get("images", []),
                "word_count": obj.get("word_count") or (len(obj.get("text","").split())),
                "feed": obj.get("feed"),
                "published": obj.get("published"),
                "scraped_at": obj.get("scraped_at", None),
                "domain": obj.get("domain") or urlparse(url).netloc,
                "meta": obj.get("meta", {})
            }

            try:
                articles_col.insert_one(doc)
                saved += 1
            except errors.DuplicateKeyError:
                skipped += 1
            except Exception as e:
                print("error inserting:", e)
                skipped += 1

    print(f"Imported {path}: total {total}, saved {saved}, skipped {skipped}")
    print("Articles collection now:", articles_col.count_documents({}))

if __name__ == "__main__":
    # default import path
    default = os.path.join(DATA, "articles.jsonl")
    import_file(default)
    # optionally add other files below or call script with different PYTHONPATH
