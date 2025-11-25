import json
from typing import Generator, Dict, Any
from pathlib import Path

def read_jsonl(path: str) -> Generator[Dict[str, Any], None, None]:
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)

def write_jsonl(path: str, items):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def append_jsonl(path: str, item: Dict[str, Any]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
