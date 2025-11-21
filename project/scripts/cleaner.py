import json
import os

BASE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(BASE, "data")

INPUT = os.path.join(DATA, "articles.jsonl")
OUTPUT = os.path.join(DATA, "articles_clean.jsonl")

seen = set()
kept = 0
dupes = 0

with open(INPUT, "r", encoding="utf-8") as infile, \
     open(OUTPUT, "w", encoding="utf-8") as outfile:

    for line in infile:
        try:
            obj = json.loads(line)
        except:
            continue

        url = obj.get("url")
        if not url:
            continue

        if url not in seen:
            seen.add(url)
            outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1
        else:
            dupes += 1

print(f"✨ CLEANED — kept: {kept}, removed: {dupes}")
