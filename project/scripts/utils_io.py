from pathlib import Path
import json

def read_jsonl(path: str | Path):
    """Reads a JSONL file and returns a list of parsed objects"""
    path = Path(path)  # ✅ Ensure string paths don’t fuck shit up
    if not path.exists():
        print(f"❌ JSONL file not found: {path}")
        return []  # return empty if file is missing instead of crying
    out = []
    try:
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"⚠️ Skipped broken JSON line: {line[:80]}...")
        print(f"✅ Loaded {len(out)} entries from {path}")
    except Exception as e:
        print(f"❌ Failed reading JSONL: {e}")
        return []
    return out


def write_jsonl(path: str | Path, docs: list[dict], mode="a"):
    """Writes a list of dicts into a JSONL file (append by default)"""
    path = Path(path)
    try:
        with open(path, mode, encoding="utf8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"✅ Wrote {len(docs)} entries → {path}")
    except Exception as e:
        print(f"❌ Failed writing JSONL: {e}")


def read_json(path: str | Path):
    """Read a normal JSON file and return the object"""
    path = Path(path)
    if not path.exists():
        print(f"❌ JSON file not found: {path}")
        return None
    try:
        with open(path, "r", encoding="utf8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed loading JSON: {e}")
        return None


def write_json(path: str | Path, obj: dict):
    """Write a dict object into a JSON file"""
    path = Path(path)
    try:
        with open(path, "w", encoding="utf8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved JSON → {path}")
    except Exception as e:
        print(f"❌ Failed writing JSON: {e}")


def ensure_dir(path: str | Path):
    """Make sure the fuckin folder exists before we save shit"""
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ Directory creation failed: {e}")


def load_existing_ids(path: str | Path):
    """Read JSON or JSONL and return a set of known IDs"""
    path = Path(path)

    if not path.exists():
        return set()

    ids = set()
    if path.suffix == ".json":
        data = read_json(path)
        if isinstance(data, list):
            for doc in data:
                if isinstance(doc, dict):
                    ids.add(doc.get("id"))
    else:
        # JSONL
        for doc in read_jsonl(path):
            ids.add(doc.get("id"))

    ids.discard(None)
    print(f"🔎 Found {len(ids)} existing IDs in {path}")
    return ids
