"""
Post-OCR text correction pipeline.

Applies multiple correction strategies:
1. SymSpell spell correction with English + automotive dictionaries
2. Dot-leader cleanup (TOC garbage)
3. Common OCR substitution map
4. Dehyphenation (line-end hyphens)
5. Confidence-weighted corrections (preserve part numbers)
6. Section code validation
"""

import os
import re
from pathlib import Path
from symspellpy import SymSpell, Verbosity

BASE_DIR = Path(r"D:\Metro Project")

# ── Singleton SymSpell instance ──────────────────────────────────────────

_symspell: SymSpell | None = None


def _get_symspell() -> SymSpell:
    """Initialize SymSpell with English frequency dictionary + automotive terms."""
    global _symspell
    if _symspell is not None:
        return _symspell

    _symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

    # Load built-in English frequency dictionary
    import symspellpy
    pkg_dir = os.path.dirname(symspellpy.__file__)
    dict_path = os.path.join(pkg_dir, "frequency_dictionary_en_82_765.txt")
    if os.path.exists(dict_path):
        _symspell.load_dictionary(dict_path, term_index=0, count_index=1)

    # Load automotive dictionary (custom terms get high frequency so they're preferred)
    auto_dict = BASE_DIR / "automotive_dictionary.txt"
    if auto_dict.exists():
        with open(auto_dict, "r") as f:
            for line in f:
                term = line.strip().lower()
                if term and not term.startswith("#"):
                    _symspell.create_dictionary_entry(term, 100000)

    return _symspell


# ── OCR Substitution Map ─────────────────────────────────────────────────
# Common OCR misreads in word context (only applied to non-part-number text)

OCR_SUBS = {
    # Character substitutions that happen frequently in scanned text
    "contro!": "control",
    "contro|": "control",
    "vehic!e": "vehicle",
    "vehic|e": "vehicle",
    "insta!!": "install",
    "insta||": "install",
    "va!ve": "valve",
    "va|ve": "valve",
    "cyc!e": "cycle",
    "cyc|e": "cycle",
    "cab!e": "cable",
    "cab|e": "cable",
    "coup!e": "couple",
    "coup|e": "couple",
    "tab!e": "table",
    "tab|e": "table",
    "possib!e": "possible",
    "possib|e": "possible",
    "availab!e": "available",
    "availab|e": "available",
    "troub!e": "trouble",
    "troub|e": "trouble",
    "assemb!y": "assembly",
    "assemb|y": "assembly",
    "supp!y": "supply",
    "supp|y": "supply",
    "app!y": "apply",
    "app|y": "apply",
    "on!y": "only",
    "on|y": "only",
    "simp!y": "simply",
    "simp|y": "simply",
    "probab!y": "probably",
    "probab|y": "probably",
    "norma!": "normal",
    "norma|": "normal",
    "manua!": "manual",
    "manua|": "manual",
    "signa!": "signal",
    "signa|": "signal",
    "meta!": "metal",
    "meta|": "metal",
    "leve!": "level",
    "leve|": "level",
    "mode!": "model",
    "mode|": "model",
    "channe!": "channel",
    "channe|": "channel",
    "fue!": "fuel",
    "fue|": "fuel",
    "oi!": "oil",
    "oi|": "oil",
    "coi!": "coil",
    "coi|": "coil",
    "rai!": "rail",
    "rai|": "rail",
    "fai!": "fail",
    "fai|": "fail",
}

# Regex-based substitutions for pipe/exclamation in word context
_PIPE_IN_WORD = re.compile(r'(?<=[a-zA-Z])[|!](?=[a-zA-Z])')
_TRAILING_PIPE = re.compile(r'(?<=[a-zA-Z])[|!]\b')


# ── Part Number Detection ────────────────────────────────────────────────

# Part numbers look like: KC0050-1B-M-RS, 96060649, 09900-21206
PART_NUMBER_RE = re.compile(
    r'\b[A-Z]{0,3}\d{3,}[-]?[A-Z0-9]*[-]?[A-Z0-9]*\b'
    r'|'
    r'\b\d{2,}-\d{2,}[-\d]*\b'
)


def _is_part_number(word: str) -> bool:
    """Check if a word looks like a part number (should not be spell-corrected)."""
    return bool(PART_NUMBER_RE.fullmatch(word))


def _is_measurement(word: str) -> bool:
    """Check if a word is a measurement value (e.g., 0.5mm, 25°C, 100kPa)."""
    return bool(re.fullmatch(r'\d+\.?\d*\s*(?:mm|cm|in|ft|lb|kg|psi|kPa|°[CF]|rpm|mph|V|A|Ω)', word))


# ── Dot-Leader Cleanup ───────────────────────────────────────────────────

# Matches dot-leader patterns from TOC pages:
# "Engine Oil........... 2-4" or "Brakes ..... ccccccceeee 3"
DOT_LEADER_RE = re.compile(r'\.{4,}[\s\S]{0,30}$', re.MULTILINE)
GARBAGE_DOTS_RE = re.compile(r'[.·•]{4,}[a-zA-Z\d\s]*$')
# Matches garbage text that follows dot leaders (OCR artifacts)
TRAILING_GARBAGE_RE = re.compile(r'\.{3,}\s*[a-z]{3,}[A-Z][a-z]*')


def clean_dot_leaders(text: str) -> str:
    """Remove dot-leader garbage from TOC pages."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        # Remove dot leaders and any garbage that follows them
        line = DOT_LEADER_RE.sub('', line)
        line = GARBAGE_DOTS_RE.sub('', line)
        line = TRAILING_GARBAGE_RE.sub('', line)
        # Remove lines that are just dots/periods
        stripped = line.strip().replace('.', '').replace(' ', '')
        if stripped or not line.strip():
            cleaned.append(line)
    return '\n'.join(cleaned)


# ── Dehyphenation ────────────────────────────────────────────────────────

HYPHEN_BREAK_RE = re.compile(r'(\w+)-\s*\n\s*(\w+)')


def dehyphenate(text: str) -> str:
    """Merge words split across lines with hyphens.

    'evapo-\\nrator' → 'evaporator'
    Only merges when the combined word is in the dictionary.
    """
    sym = _get_symspell()

    def _try_merge(m: re.Match) -> str:
        part1 = m.group(1)
        part2 = m.group(2)
        combined = part1 + part2
        hyphenated = part1 + "-" + part2

        # Check if combined form is a known word
        suggestions = sym.lookup(combined.lower(), Verbosity.TOP, max_edit_distance=0)
        if suggestions:
            return combined

        # Check if hyphenated form is intentional (e.g., "self-test")
        suggestions = sym.lookup(hyphenated.lower(), Verbosity.TOP, max_edit_distance=0)
        if suggestions:
            return hyphenated

        # If combined word looks plausible via edit distance, merge
        suggestions = sym.lookup(combined.lower(), Verbosity.CLOSEST, max_edit_distance=1)
        if suggestions:
            return suggestions[0].term

        # Keep the hyphenated line break as-is
        return hyphenated + "\n"

    return HYPHEN_BREAK_RE.sub(_try_merge, text)


# ── OCR Character Fixes ─────────────────────────────────────────────────

def fix_ocr_chars(text: str) -> str:
    """Apply known OCR substitution fixes."""
    # Apply direct word substitutions
    for bad, good in OCR_SUBS.items():
        if bad in text.lower():
            # Case-preserving replacement
            pattern = re.compile(re.escape(bad), re.IGNORECASE)
            text = pattern.sub(good, text)

    # Fix pipe/exclamation used as 'l' within words
    def _replace_pipe_in_word(m: re.Match) -> str:
        return 'l'

    text = _PIPE_IN_WORD.sub(_replace_pipe_in_word, text)
    text = _TRAILING_PIPE.sub('l', text)

    return text


# ── Section Code Validation ─────────────────────────────────────────────

SECTION_CODE_FIX_RE = re.compile(r'\b([O0-9]+[A-Z]\d*)-(\d+)\b')


def fix_section_codes(text: str) -> str:
    """Fix section codes: O→0, l→1 in section-page references."""
    def _fix(m: re.Match) -> str:
        code = m.group(1)
        page = m.group(2)
        # O → 0 in the code portion
        code = code.replace('O', '0')
        # l → 1 in the page number portion
        page = page.replace('l', '1')
        return f"{code}-{page}"

    return SECTION_CODE_FIX_RE.sub(_fix, text)


# ── SymSpell Correction ─────────────────────────────────────────────────

def spell_correct_text(text: str, confidence_data: dict | None = None) -> str:
    """Apply SymSpell correction to low-confidence words.

    Args:
        text: The OCR text to correct.
        confidence_data: Optional dict mapping words to confidence scores (0-100).
            If provided, only correct words with confidence < 80.
    """
    sym = _get_symspell()
    words = text.split()
    corrected = []

    for word in words:
        # Strip punctuation for lookup but preserve it
        stripped = word.strip('.,;:!?()[]{}"\'-')
        prefix = word[:word.index(stripped)] if stripped and stripped in word else ''
        suffix = word[word.index(stripped) + len(stripped):] if stripped and stripped in word else ''

        if not stripped:
            corrected.append(word)
            continue

        # Skip part numbers, measurements, and very short words
        if _is_part_number(stripped) or _is_measurement(stripped) or len(stripped) < 3:
            corrected.append(word)
            continue

        # Skip words that are all uppercase (acronyms, headers)
        if stripped.isupper() and len(stripped) > 1:
            corrected.append(word)
            continue

        # If confidence data available, skip high-confidence words
        if confidence_data and stripped in confidence_data:
            if confidence_data[stripped] >= 80:
                corrected.append(word)
                continue

        # Try spell correction
        suggestions = sym.lookup(stripped.lower(), Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            best = suggestions[0]
            # Only correct if edit distance is small relative to word length
            if best.distance > 0 and best.distance <= max(1, len(stripped) // 3):
                # Preserve original capitalization pattern
                if stripped[0].isupper():
                    fixed = best.term.capitalize()
                else:
                    fixed = best.term
                corrected.append(prefix + fixed + suffix)
            else:
                corrected.append(word)
        else:
            corrected.append(word)

    return ' '.join(corrected)


# ── Main Entry Point ─────────────────────────────────────────────────────

def correct_text(text: str, confidence_data: dict | None = None,
                 is_toc: bool = False) -> str:
    """Apply full post-OCR correction pipeline to text.

    Args:
        text: Raw OCR text.
        confidence_data: Optional per-word confidence scores.
        is_toc: If True, applies dot-leader cleanup.

    Returns:
        Corrected text.
    """
    if not text or not text.strip():
        return text

    # 1. Fix known OCR character substitutions
    text = fix_ocr_chars(text)

    # 2. Clean dot-leader garbage (TOC pages)
    if is_toc:
        text = clean_dot_leaders(text)

    # 3. Fix section codes
    text = fix_section_codes(text)

    # 4. Dehyphenate split words
    text = dehyphenate(text)

    # 5. Spell correction (last, so it works on cleaned text)
    text = spell_correct_text(text, confidence_data)

    return text
