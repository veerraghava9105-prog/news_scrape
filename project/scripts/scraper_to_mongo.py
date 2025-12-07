# scraper_to_mongo.py
import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os
import sys
from pymongo import MongoClient, errors
from tqdm import tqdm

# -----------------------------
# PROJECT PATHS - adjust if needed
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # project/
DATA = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA, exist_ok=True)

QUEUE = os.path.join(DATA, "queue.jsonl")
SCRAPED = os.path.join(DATA, "scraped_urls.jsonl")

# -----------------------------
# MONGO SETUP
# -----------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = mongo.newsdb
articles_col = db.articles
failed_col = db.failed_articles

# unique index for safe dedupe
articles_col.create_index("url", unique=True)
failed_col.create_index("url", unique=True)

# -----------------------------
# Helper: scraped URL store
# -----------------------------
def load_scraped():
    s = set()
    if os.path.exists(SCRAPED):
        with open(SCRAPED, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    s.add(json.loads(line)["url"])
                except:
                    pass
    return s

def add_scraped(url):
    with open(SCRAPED, "a", encoding="utf-8") as f:
        f.write(json.dumps({"url": url}) + "\n")

# -----------------------------
# Extract text & soup
# -----------------------------
BAD_PHRASES = [
    "Sign up for", "Subscribe", "Read our review", "May earn a commission",
    "Continue reading", "No comments yet", "Comments", "Follow us on", "Related Articles"
]

def clean_boilerplate(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    out = []
    seen = set()
    for line in lines:
        if any(p.lower() in line.lower() for p in BAD_PHRASES):
            continue
        # dedupe repeated identical lines (common on some sites)
        if lines.count(line) > 1:
            if line in seen:
                continue
            seen.add(line)
        out.append(line)
    return "\n".join(out)

def fetch_html(url):
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        return r.text
    except Exception as e:
        return None

def extract_text_and_soup(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
        tag.decompose()
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    cleaned = [p for p in paras if len(p.split()) > 5]
    if len(cleaned) < 3:
        return None, soup
    text = "\n".join(cleaned)
    text = clean_boilerplate(text)
    return text, soup

# -----------------------------
# Smart image extraction
# -----------------------------
def extract_images(soup, max_keep=3):
    if soup is None:
        return []
    blacklist = ["logo", "icon", "svg", "pixel", "analytics", "scorecard",
                 "facebook", "reddit", "share", "twitter", "spacer", "gif", "ad", "banner", "google"]
    good_ext = (".jpg", ".jpeg", ".png", ".webp", ".avif")

    found = []
    for im in soup.find_all("img"):
        src = im.get("src") or im.get("data-src") or im.get("data-original") or im.get("data-lazy")
        if not src:
            continue
        src = src.strip()
        if not (src.startswith("http://") or src.startswith("https://")):
            # sometimes relative URLs, try to skip (we could join with base but skip for speed)
            continue
        low = src.lower()
        # filter by extension & blacklist
        if not any(low.endswith(ext) for ext in good_ext):
            # allow if it contains .jpg?param= but endswith doesn't catch — quick check:
            if not any(ext in low for ext in good_ext):
                continue
        if any(b in low for b in blacklist):
            continue
        found.append(src)

    # dedupe while preserving order
    dedup = []
    seen = set()
    for u in found:
        if u in seen:
            continue
        seen.add(u)
        dedup.append(u)

    # choose up to max_keep images
    out = []
    for i, url in enumerate(dedup[:max_keep]):
        out.append({
            "source_url": url,
            "priority": i+1,
            "status": "pending",
            "attempts": 0,
            "local_path": None,
            "s3_url": None,
            "bytes": None
        })
    return out

# -----------------------------
# Main scraper
# -----------------------------
def scrape():
    print("\n🔥 SCRAPER STARTED (Mongo mode)\n")
    scraped_urls = load_scraped()
    saved = 0
    failed = 0
    total = 0

    if not os.path.exists(QUEUE):
        print("Queue file not found:", QUEUE)
        return

    with open(QUEUE, "r", encoding="utf-8") as q:
        lines = q.readlines()

    for line in lines:
        total += 1
        item = json.loads(line)
        url = item.get("link") or item.get("url")
        if not url:
            continue
        if url in scraped_urls:
            # skip silently (already scraped)
            continue

        print(f"\n→ Scraping: {url}")
        html = fetch_html(url)
        if not html:
            print("   ❌ REQUEST FAILED")
            failed_col.update_one({"url": url}, {"$set":{"reason":"request_failed","feed":item.get("feed")}}, upsert=True)
            # DO NOT mark as scraped, try again later
            failed += 1
            continue

        text, soup = extract_text_and_soup(html)
        images = extract_images(soup)

        if not text or len(text.split()) < 80:
            print("   ❌ TOO SHORT — NOT SAVED")
            failed_col.update_one({"url": url}, {"$set":{
                "reason":"too_short", "title": item.get("title"), "feed": item.get("feed"), "sample": (text or "")[:200]
            }}, upsert=True)
            # mark as scraped to avoid thrashing; we assume it's not useful
            add_scraped(url)
            scraped_urls.add(url)
            failed += 1
            continue

        doc = {
            "url": url,
            "title": item.get("title"),
            "text": text,
            "images": images,
            "word_count": len(text.split()),
            "feed": item.get("feed"),
            "published": item.get("published"),
            "scraped_at": time.time(),
            "domain": urlparse(url).netloc,
            "meta": {}
        }

        try:
            articles_col.insert_one(doc)
            print(f"   ✔ SAVED ({doc['word_count']} words, images: {len(images)})")
            saved += 1
            add_scraped(url)
            scraped_urls.add(url)
        except errors.DuplicateKeyError:
            print("   ⚠ ALREADY IN DB (duplicate)")
            # mark scraped anyway
            add_scraped(url)
            scraped_urls.add(url)
            continue
        except Exception as e:
            print("   ❌ MONGO INSERT FAILED", str(e))
            failed_col.update_one({"url": url}, {"$set":{"reason":"mongo_insert_failed","error":str(e)}}, upsert=True)
            failed += 1

    # clear queue after processing
    open(QUEUE, "w").close()
    print("\n🧹 queue cleared!")
    print(f"\n🎉 DONE — processed: {total}, saved: {saved}, failed: {failed}")
    print("Failed collection size:", failed_col.count_documents({}))
    print("Articles collection size:", articles_col.count_documents({}))

if __name__ == "__main__":
    scrape()
