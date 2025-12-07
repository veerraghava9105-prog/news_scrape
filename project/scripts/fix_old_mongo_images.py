# fix_old_articles_add_images.py
import os
import json
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from urllib.parse import urljoin, urlparse
from datetime import datetime
from tqdm import tqdm

# mongo connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "newsdb"
COLLECTION = "articles"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
col = db[COLLECTION]

HEADERS = {"User-Agent": "Mozilla/5.0 (OldFixer/1.0)"}
TIMEOUT = 10
MAX_IMAGES = 6

# bad image patterns
BAD = ["logo", "icon", "svg", "pixel", "1x1", "facebook", "twitter",
       "reddit", "spacer", "gif", "tracking"]

GOOD_EXT = (".jpg", ".jpeg", ".png", ".webp", ".avif")

def normalize_url(base, src):
    if not src: return None
    src = src.strip()
    if src.startswith("data:") or src.startswith("javascript:"):
        return None
    if src.startswith("//"):
        return "https:" + src
    if src.startswith("http://") or src.startswith("https://"):
        return src
    return urljoin(base, src)

def is_good_image(url):
    if not url: return False
    low = url.lower()
    if not any(ext in low for ext in GOOD_EXT):
        return False
    if any(b in low for b in BAD):
        return False
    return True


def extract_images_from_html(url, html):
    soup = BeautifulSoup(html, "html.parser")
    candidates = []

    # 1) OG images
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        u = normalize_url(url, og["content"])
        if u: candidates.append(u)

    tw = soup.find("meta", {"name": "twitter:image"})
    if tw and tw.get("content"):
        u = normalize_url(url, tw["content"])
        if u: candidates.append(u)

    # 2) LD+JSON
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "{}")
            if isinstance(data, dict):
                if isinstance(data.get("image"), str):
                    candidates.append(normalize_url(url, data["image"]))
                elif isinstance(data.get("image"), list):
                    for it in data["image"]:
                        candidates.append(normalize_url(url, it))
        except:
            pass

    # 3) Extract <img> from article/main tags
    article_nodes = soup.select("article, main, .article, .entry-content, .story-body")
    for node in article_nodes:
        for img in node.find_all("img"):
            src = img.get("src") or img.get("data-src")
            u = normalize_url(url, src)
            if u: candidates.append(u)

    # 4) fallback: every img
    if not candidates:
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src")
            u = normalize_url(url, src)
            if u: candidates.append(u)

    # filter
    final = []
    for u in candidates:
        if u and is_good_image(u) and u not in final:
            final.append(u)
        if len(final) >= MAX_IMAGES:
            break

    return final


def fix_old():
    docs = list(col.find({"images": {"$size": 0}}))  # only old docs
    print(f"\nFound {len(docs)} old articles missing images.\n")

    for doc in tqdm(docs):
        url = doc["url"]

        try:
            res = requests.get(url, timeout=TIMEOUT, headers=HEADERS)
            html = res.text
        except:
            continue

        urls = extract_images_from_html(url, html)
        if not urls:
            continue

        formatted_images = []
        for i, img in enumerate(urls):
            formatted_images.append({
                "source_url": img,
                "priority": i + 1,
                "status": "pending",
                "attempts": 0,
                "local_path": None,
                "s3_url": None,
                "bytes": None
            })

        col.update_one(
            {"_id": doc["_id"]},
            {"$set": {"images": formatted_images}}
        )

    print("\n🎉 Image fixing complete!\n")


if __name__ == "__main__":
    fix_old()
