"""Comprehensive statistical audit of build/chunks.jsonl"""
import sys, json, os
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), '..', 'build', 'chunks.jsonl')

# ── Load ──────────────────────────────────────────────────────────────
chunks = []
with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            chunks.append(json.loads(line))

print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}\n")

# ── Helpers ───────────────────────────────────────────────────────────
def percentile(sorted_vals, p):
    """Simple nearest-rank percentile."""
    if not sorted_vals:
        return 0
    k = int(len(sorted_vals) * p / 100)
    k = min(k, len(sorted_vals) - 1)
    return sorted_vals[k]

def stats_row(vals):
    vals = sorted(vals)
    n = len(vals)
    if n == 0:
        return {}
    return {
        'count': n,
        'min': vals[0],
        'p5': percentile(vals, 5),
        'p25': percentile(vals, 25),
        'median': percentile(vals, 50),
        'p75': percentile(vals, 75),
        'p95': percentile(vals, 95),
        'max': vals[-1],
        'mean': round(sum(vals) / n, 1),
    }

# ══════════════════════════════════════════════════════════════════════
# 1. Overall stats
# ══════════════════════════════════════════════════════════════════════
print("=" * 80)
print("1. OVERALL STATS")
print("=" * 80)

type_counts = Counter(c.get('type', '<none>') for c in chunks)
token_counts = [c.get('token_count', 0) for c in chunks]
overall = stats_row(token_counts)

print(f"\nTotal chunks: {len(chunks)}")
print(f"\nChunks by type:")
for t, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {t:25s}  {cnt:5d}")

print(f"\nToken distribution (all chunks):")
for k, v in overall.items():
    print(f"  {k:8s}: {v}")

# ══════════════════════════════════════════════════════════════════════
# 2. Token distribution by type
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("2. TOKEN DISTRIBUTION BY TYPE")
print("=" * 80)

by_type = defaultdict(list)
for c in chunks:
    by_type[c.get('type', '<none>')].append(c.get('token_count', 0))

header = f"{'type':25s} {'count':>6s} {'min':>5s} {'p25':>5s} {'med':>5s} {'p75':>5s} {'max':>6s} {'mean':>7s}"
print(f"\n{header}")
print("-" * len(header))
for t in sorted(by_type.keys()):
    s = stats_row(by_type[t])
    print(f"{t:25s} {s['count']:6d} {s['min']:5d} {s['p25']:5d} {s['median']:5d} {s['p75']:5d} {s['max']:6d} {s['mean']:7.1f}")

# ══════════════════════════════════════════════════════════════════════
# 3. Tiny chunks (<30 tokens)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("3. TINY CHUNKS (< 30 tokens)")
print("=" * 80)

tiny = [c for c in chunks if c.get('token_count', 0) < 30]
tiny_by_type = Counter(c.get('type', '<none>') for c in tiny)
print(f"\nTotal tiny chunks: {len(tiny)}")
print(f"\nTiny by type:")
for t, cnt in sorted(tiny_by_type.items(), key=lambda x: -x[1]):
    print(f"  {t:25s}  {cnt:5d}")

print(f"\n15 examples of tiny chunks:")
for c in tiny[:15]:
    text_preview = (c.get('text') or '').replace('\n', ' ')[:100]
    print(f"  [{c.get('type','?'):15s}] tokens={c.get('token_count',0):3d}  pg={str(c.get('page','')):>4s}  | {text_preview}")

# ══════════════════════════════════════════════════════════════════════
# 4. Huge chunks (>500 tokens)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("4. HUGE CHUNKS (> 500 tokens)")
print("=" * 80)

huge = [c for c in chunks if c.get('token_count', 0) > 500]
huge_by_type = Counter(c.get('type', '<none>') for c in huge)
huge_sorted = sorted(huge, key=lambda c: -c.get('token_count', 0))

print(f"\nTotal huge chunks: {len(huge)}")
print(f"\nHuge by type:")
for t, cnt in sorted(huge_by_type.items(), key=lambda x: -x[1]):
    print(f"  {t:25s}  {cnt:5d}")

print(f"\nTop 10 largest chunks:")
for c in huge_sorted[:10]:
    text_preview = (c.get('text') or '').replace('\n', ' ')[:120]
    print(f"  chunk_id={c.get('chunk_id','?'):30s}  pg={str(c.get('page','')):>4s}  type={c.get('type','?'):15s}  tokens={c.get('token_count',0):5d}")
    print(f"    {text_preview}")

# ══════════════════════════════════════════════════════════════════════
# 5. Empty or near-empty text
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("5. EMPTY OR NEAR-EMPTY TEXT")
print("=" * 80)

empty = [c for c in chunks if not (c.get('text') or '').strip()]
near_empty = [c for c in chunks if 0 < len((c.get('text') or '').strip()) <= 5]

print(f"\nChunks with empty/whitespace-only text: {len(empty)}")
for c in empty[:20]:
    print(f"  chunk_id={c.get('chunk_id','?'):30s}  type={c.get('type','?'):15s}  pg={c.get('page','')}")

print(f"\nChunks with text <= 5 chars (near-empty): {len(near_empty)}")
for c in near_empty[:20]:
    text = (c.get('text') or '').strip()
    print(f"  chunk_id={c.get('chunk_id','?'):30s}  type={c.get('type','?'):15s}  pg={c.get('page','')}  text={repr(text)}")

# ══════════════════════════════════════════════════════════════════════
# 6. Duplicate text detection
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("6. DUPLICATE TEXT DETECTION (exact)")
print("=" * 80)

text_map = defaultdict(list)
for c in chunks:
    txt = (c.get('text') or '').strip()
    if txt:
        text_map[txt].append(c)

dupes = {txt: clist for txt, clist in text_map.items() if len(clist) > 1}
total_dupe_chunks = sum(len(v) for v in dupes.values())
print(f"\nDistinct texts appearing more than once: {len(dupes)}")
print(f"Total chunks involved in duplicates:    {total_dupe_chunks}")

# Show up to 15 duplicate groups
shown = 0
for txt, clist in sorted(dupes.items(), key=lambda x: -len(x[1])):
    if shown >= 15:
        break
    shown += 1
    preview = txt.replace('\n', ' ')[:100]
    print(f"\n  [{len(clist)}x] \"{preview}\"")
    for c in clist[:5]:
        print(f"       chunk_id={c.get('chunk_id','?'):30s}  type={c.get('type','?'):15s}  pg={c.get('page','')}")
    if len(clist) > 5:
        print(f"       ... and {len(clist)-5} more")

# ══════════════════════════════════════════════════════════════════════
# 7. Section coverage
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("7. SECTION COVERAGE (by section_code)")
print("=" * 80)

sec_counts = Counter()
sec_types = defaultdict(Counter)
for c in chunks:
    sc = c.get('section_code') or '<none>'
    sec_counts[sc] += 1
    sec_types[sc][c.get('type', '<none>')] += 1

print(f"\nTotal distinct section_codes: {len(sec_counts)}")
print(f"\n{'section_code':>14s} {'count':>6s}   types")
print("-" * 70)
for sc, cnt in sorted(sec_counts.items(), key=lambda x: x[0]):
    types_str = ", ".join(f"{t}:{n}" for t, n in sorted(sec_types[sc].items(), key=lambda x: -x[1]))
    marker = " *** LOW" if 0 < cnt < 3 and sc != '<none>' else ""
    print(f"  {sc:>12s} {cnt:6d}   {types_str}{marker}")

low_sections = {sc: cnt for sc, cnt in sec_counts.items() if 0 < cnt < 3 and sc != '<none>'}
print(f"\nSections with < 3 chunks (potentially missing content): {len(low_sections)}")
for sc, cnt in sorted(low_sections.items()):
    print(f"  {sc}: {cnt} chunk(s)")

# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"  Total chunks:          {len(chunks)}")
print(f"  Distinct types:        {len(type_counts)}")
print(f"  Token range:           {overall['min']} - {overall['max']}  (median {overall['median']}, mean {overall['mean']})")
print(f"  Tiny (<30 tokens):     {len(tiny)}  ({100*len(tiny)/len(chunks):.1f}%)")
print(f"  Huge (>500 tokens):    {len(huge)}  ({100*len(huge)/len(chunks):.1f}%)")
print(f"  Empty text:            {len(empty)}")
print(f"  Near-empty (<=5ch):    {len(near_empty)}")
print(f"  Duplicate groups:      {len(dupes)} groups, {total_dupe_chunks} chunks")
print(f"  Section codes:         {len(sec_counts)} (low coverage: {len(low_sections)})")
print()
