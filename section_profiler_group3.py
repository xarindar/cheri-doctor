#!/usr/bin/env python3
"""
Section Profiler — Build router map profiles for Group 3 sections.
Analyzes chunk text to find distinctive terms, confusion terms, and section neighborhoods.
Group 3: General, Maintenance, HVAC, Body, Wiring
"""

import json
import re
import math
from collections import Counter, defaultdict
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
TARGET_SECTIONS = {"0A", "0B", "1A", "1B", "1D", "9J", "10",
                   "10-1", "10-2", "10-3", "10-4", "10-5", "10-6",
                   "10-7", "10-8", "10-9", "10-10", "10-11"}

CHUNK_FILES = [
    Path("/home/abe/metro-project/build/chunks.jsonl"),
    Path("/home/abe/metro-project/build_supplement/chunks.jsonl"),
]

# Common English stop words + manual boilerplate
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "shall", "should", "may", "might", "must", "can", "could",
    "not", "no", "nor", "so", "if", "then", "than", "that", "this",
    "these", "those", "it", "its", "i", "we", "you", "he", "she", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "our", "their",
    "which", "who", "whom", "what", "where", "when", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "only",
    "own", "same", "also", "into", "over", "after", "before", "between",
    "through", "during", "above", "below", "up", "down", "out", "off",
    "about", "again", "further", "once", "here", "there", "any", "just",
    "too", "very", "much", "well", "back", "even", "still", "way",
    # Manual boilerplate
    "figure", "fig", "section", "see", "refer", "note", "page", "shown",
    "install", "remove", "replace", "using", "use", "used", "check",
    "make", "sure", "step", "following", "procedure", "perform",
    "service", "manual", "metro", "geo", "vehicle", "car",
}

# Known multi-word phrases to detect (lowercased) — Group 3 relevant
PHRASES = [
    # General / Maintenance (0A, 0B)
    "oil change", "engine oil", "oil filter", "spark plug", "air filter",
    "tire rotation", "tire pressure", "wheel alignment", "fluid level",
    "maintenance schedule", "scheduled maintenance", "inspection",
    "coolant flush", "brake fluid", "transmission fluid", "power steering fluid",
    "drive belt", "serpentine belt", "timing belt",
    "vin", "vehicle identification", "serial number", "model year",
    "specifications", "torque specifications", "capacities",
    "jacking", "jack stand", "lifting", "towing",
    # HVAC (1A, 1B, 1D)
    "heater core", "blower motor", "blower fan", "hvac", "air conditioning",
    "a/c", "compressor", "condenser", "evaporator", "refrigerant",
    "expansion valve", "orifice tube", "receiver drier", "accumulator",
    "heater hose", "heater control", "temperature control", "mode door",
    "defrost", "defroster", "fresh air", "recirculation",
    "compressor clutch", "compressor belt", "refrigerant charge",
    "sight glass", "low pressure", "high pressure", "suction", "discharge",
    # Body (9J, 10, 10-x)
    "door panel", "door handle", "door lock", "window regulator",
    "windshield", "rear window", "side window", "glass",
    "seat belt", "seat track", "seat cushion", "headrest",
    "bumper", "fender", "hood", "trunk", "hatch", "liftgate",
    "instrument panel", "dashboard", "glove box", "console",
    "trim panel", "garnish", "molding", "weatherstrip", "seal",
    "wiper motor", "wiper arm", "wiper blade", "washer",
    "mirror", "sun visor", "headliner", "carpet", "floor mat",
    "body panel", "quarter panel", "rocker panel", "pillar",
    "hinge", "latch", "striker", "lock cylinder",
    # Wiring
    "wiring diagram", "wiring harness", "wire color", "connector",
    "junction block", "fuse block", "fuse", "relay",
    "ground point", "chassis ground", "circuit", "terminal",
]


def load_chunks():
    """Load all chunks, grouped by section_code."""
    sections = defaultdict(list)
    all_sections = defaultdict(list)
    for fpath in CHUNK_FILES:
        with open(fpath) as f:
            for line in f:
                chunk = json.loads(line)
                code = chunk.get("section_code")
                if not code:
                    continue
                text = chunk.get("text", "")
                if len(text.strip()) < 20:
                    continue
                all_sections[code].append(text)
                if code in TARGET_SECTIONS:
                    sections[code].append(text)
    return sections, all_sections


def tokenize(text):
    """Tokenize text into unigrams and bigrams, filtering stop words."""
    text = text.lower()
    # Extract words (keep alphanumeric and hyphens)
    words = re.findall(r"[a-z][a-z0-9\-]*(?:'[a-z]+)?", text)
    # Filter stop words for unigrams
    unigrams = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    # Build bigrams from non-stop words in sequence
    bigrams = []
    for i in range(len(words) - 1):
        if words[i] not in STOP_WORDS and words[i + 1] not in STOP_WORDS:
            bigrams.append(f"{words[i]} {words[i+1]}")
    # Also extract trigrams for known phrases
    trigrams = []
    for i in range(len(words) - 2):
        tri = f"{words[i]} {words[i+1]} {words[i+2]}"
        trigrams.append(tri)
    return unigrams, bigrams, trigrams


def build_term_freqs(sections):
    """Build term frequency counters per section."""
    tf = {}
    doc_counts = {}  # how many sections contain each term
    total_term_count = Counter()

    for code, texts in sections.items():
        counter = Counter()
        for text in texts:
            unigrams, bigrams, trigrams = tokenize(text)
            seen_in_doc = set()
            for term in unigrams:
                counter[term] += 1
                seen_in_doc.add(term)
            for term in bigrams:
                counter[term] += 1
                seen_in_doc.add(term)
            for term in trigrams:
                counter[term] += 1
                seen_in_doc.add(term)
            for term in seen_in_doc:
                total_term_count[term] += 1
        tf[code] = counter

    # Count how many sections each term appears in
    term_sections = defaultdict(set)
    for code, counter in tf.items():
        for term in counter:
            term_sections[term].add(code)

    return tf, term_sections


def compute_tfidf(tf, term_sections, n_sections):
    """Compute TF-IDF scores per section."""
    tfidf = {}
    for code, counter in tf.items():
        total = sum(counter.values())
        scores = {}
        for term, count in counter.items():
            tf_score = count / total
            df = len(term_sections[term])
            idf = math.log(n_sections / (1 + df))
            scores[term] = tf_score * idf
        tfidf[code] = scores
    return tfidf


def find_section_names(all_sections):
    """Try to extract a readable name for each section from section_path or first chunk."""
    # We need to re-read chunks to get section_path
    names = {}
    for fpath in CHUNK_FILES:
        with open(fpath) as f:
            for line in f:
                chunk = json.loads(line)
                code = chunk.get("section_code")
                if code and code not in names:
                    sp = chunk.get("section_path", "")
                    if sp and sp != "Unknown":
                        names[code] = sp
    return names


def main():
    print("=" * 80)
    print("SECTION PROFILER — Group 3 (General, Maintenance, HVAC, Body, Wiring)")
    print("=" * 80)

    # Load
    sections, all_sections = load_chunks()
    section_names = find_section_names(all_sections)

    print(f"\nLoaded {sum(len(v) for v in sections.values())} chunks across {len(sections)} target sections")
    print(f"Total sections in corpus: {len(all_sections)}")

    for code in sorted(TARGET_SECTIONS):
        n = len(sections.get(code, []))
        name = section_names.get(code, "?")
        print(f"  {code:6s}: {n:4d} chunks  — {name}")

    # Build TF over ALL sections (for IDF calculation)
    tf_all, term_sections_all = build_term_freqs(all_sections)
    n_all = len(all_sections)

    # Build TF over target sections only (for confusion analysis)
    tf_target, term_sections_target = build_term_freqs(sections)
    n_target = len(sections)

    # Compute TF-IDF using ALL sections for IDF (more discriminative)
    tfidf_all = compute_tfidf(tf_all, term_sections_all, n_all)

    # ── Per-section profiles ────────────────────────────────────────────
    for code in sorted(TARGET_SECTIONS):
        if code not in sections:
            print(f"\n{'='*80}")
            print(f"SECTION {code}: NO CHUNKS FOUND")
            continue

        name = section_names.get(code, "Unknown")
        scores = tfidf_all.get(code, {})
        tf_code = tf_all.get(code, Counter())
        total_terms = sum(tf_code.values())

        print(f"\n{'='*80}")
        print(f"SECTION {code}: {name}")
        print(f"  Chunks: {len(sections[code])}, Total term occurrences: {total_terms}")
        print(f"{'='*80}")

        # ── Top distinctive terms (high TF-IDF) ────────────────────────
        # Filter: must appear at least 3 times, skip very short terms
        candidates = {t: s for t, s in scores.items()
                      if tf_code[t] >= 3 and len(t) > 2}

        # Separate unigrams and multi-word
        unigram_scores = {t: s for t, s in candidates.items() if " " not in t}
        multiword_scores = {t: s for t, s in candidates.items() if " " in t}

        top_unigrams = sorted(unigram_scores.items(), key=lambda x: -x[1])[:20]
        top_multiword = sorted(multiword_scores.items(), key=lambda x: -x[1])[:15]

        print(f"\n  DISTINCTIVE TERMS (top unigrams by TF-IDF):")
        for term, score in top_unigrams:
            n_sects = len(term_sections_all[term])
            freq = tf_code[term]
            print(f"    {term:30s}  freq={freq:4d}  sections={n_sects:2d}  tfidf={score:.6f}")

        print(f"\n  DISTINCTIVE PHRASES (top multi-word by TF-IDF):")
        for term, score in top_multiword:
            n_sects = len(term_sections_all[term])
            freq = tf_code[term]
            print(f"    {term:40s}  freq={freq:4d}  sections={n_sects:2d}  tfidf={score:.6f}")

        # ── Confusion terms ─────────────────────────────────────────────
        # Terms that are frequent in THIS section AND in other target sections
        print(f"\n  CONFUSION TERMS (shared with other Group 3 sections):")
        confusion = []
        for term, freq in tf_code.most_common(500):
            if len(term) <= 2:
                continue
            # Which other TARGET sections also have this term frequently?
            other_sections = []
            for other_code in sorted(TARGET_SECTIONS):
                if other_code == code:
                    continue
                other_tf = tf_all.get(other_code, Counter())
                other_freq = other_tf.get(term, 0)
                if other_freq >= 3:
                    other_sections.append((other_code, other_freq))
            if other_sections:
                confusion.append((term, freq, other_sections))

        # Sort by number of overlapping sections (desc), then by freq
        confusion.sort(key=lambda x: (-len(x[2]), -x[1]))
        shown = 0
        for term, freq, others in confusion[:25]:
            others_str = ", ".join(f"{c}({f})" for c, f in others[:6])
            print(f"    {term:35s}  here={freq:4d}  also_in: {others_str}")
            shown += 1

        # ── Section neighborhood ────────────────────────────────────────
        print(f"\n  SECTION NEIGHBORHOOD (most term overlap):")
        overlap_scores = Counter()
        for term, freq in tf_code.items():
            if freq < 2:
                continue
            for other_code in term_sections_all[term]:
                if other_code != code and other_code in TARGET_SECTIONS:
                    # Weight by min frequency in both sections
                    other_freq = tf_all[other_code].get(term, 0)
                    overlap_scores[other_code] += min(freq, other_freq)

        for neighbor, score in overlap_scores.most_common(8):
            neighbor_name = section_names.get(neighbor, "?")
            print(f"    {neighbor:6s} (overlap={score:5d})  {neighbor_name}")

    # ── Global confusion matrix (summary) ───────────────────────────────
    print(f"\n{'='*80}")
    print("GLOBAL CONFUSION MATRIX — Top shared terms between section pairs")
    print(f"{'='*80}")

    pairs_seen = set()
    pair_overlaps = []
    for c1 in sorted(TARGET_SECTIONS):
        for c2 in sorted(TARGET_SECTIONS):
            if c1 >= c2:
                continue
            pair = (c1, c2)
            if pair in pairs_seen:
                continue
            pairs_seen.add(pair)

            tf1 = tf_all.get(c1, Counter())
            tf2 = tf_all.get(c2, Counter())
            shared_terms = set(tf1.keys()) & set(tf2.keys())
            # Find terms that are reasonably frequent in both
            shared_significant = []
            for t in shared_terms:
                if tf1[t] >= 3 and tf2[t] >= 3 and len(t) > 2:
                    shared_significant.append((t, tf1[t], tf2[t]))
            if not shared_significant:
                continue

            shared_significant.sort(key=lambda x: -(x[1] + x[2]))
            total_overlap = sum(min(a, b) for _, a, b in shared_significant)
            pair_overlaps.append((c1, c2, total_overlap, shared_significant[:8]))

    pair_overlaps.sort(key=lambda x: -x[2])
    for c1, c2, total, top_terms in pair_overlaps[:30]:
        n1 = section_names.get(c1, "?")
        n2 = section_names.get(c2, "?")
        terms_str = ", ".join(f"{t}({a}/{b})" for t, a, b in top_terms[:5])
        print(f"\n  {c1} <-> {c2}  (overlap={total})")
        print(f"    {c1}: {n1}")
        print(f"    {c2}: {n2}")
        print(f"    Top shared: {terms_str}")


if __name__ == "__main__":
    main()
