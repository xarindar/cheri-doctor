import json
from pathlib import Path
from collections import Counter

def inspect_index():
    index_dir = Path("tools/rag_index")
    lookup_path = index_dir / "chunk_lookup.json"
    
    if not lookup_path.exists():
        print("Index lookup file not found.")
        return

    with open(lookup_path, "r", encoding="utf-8") as f:
        lookup = json.load(f)

    print(f"Total chunks in index: {len(lookup)}")
    
    page_counts = Counter()
    systems = Counter()
    
    target_pages_chunks = []

    for cid, chunk in lookup.items():
        page = chunk.get("page")
        if page is not None:
            page_counts[page] += 1
            if 130 <= page <= 140:
                target_pages_chunks.append(chunk)
        
        sys = chunk.get("system")
        if sys:
            systems[sys] += 1

    print("\nChunks per page (for pages 130-140):")
    for p in range(130, 141):
        count = page_counts.get(p, 0)
        print(f"  Page {p}: {count}")

    print("\nSample chunks from 130-140:")
    for c in target_pages_chunks[:5]:
        print(f"  [{c['chunk_id']}] p{c['page']} type={c.get('type')} system={c.get('system')}")

    print("\nSystem distribution:")
    for s, c in systems.most_common(10):
        print(f"  {s}: {c}")

if __name__ == "__main__":
    inspect_index()
