"""
pipeline/extract/text_cleaner.py
Normalize and clean text extracted from financial PDFs.
Handles common PDF artifacts: ligatures, encoding issues, spacing noise.
"""

import re
import unicodedata


# ─────────────────────────────────────────────
# Ligature and encoding fixes
# ─────────────────────────────────────────────
LIGATURES = {
    "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
    "\ufb00": "ff", "\u2019": "'", "\u2018": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-",
    "\u00a0": " ",  # non-breaking space
    "\u200b": "",   # zero-width space
    "\ufffd": "",   # replacement char
}

LIGATURE_TABLE = str.maketrans(LIGATURES)


def fix_ligatures(text: str) -> str:
    return text.translate(LIGATURE_TABLE)


# ─────────────────────────────────────────────
# Common financial PDF noise patterns
# ─────────────────────────────────────────────
NOISE_PATTERNS = [
    (re.compile(r"Page \d+ of \d+", re.IGNORECASE), ""),
    (re.compile(r"^\s*\d+\s*$", re.MULTILINE), ""),       # lone page numbers
    (re.compile(r"CONFIDENTIAL.*?$", re.IGNORECASE | re.MULTILINE), ""),
    (re.compile(r"(?:www\.|http)[^\s]+"), ""),              # URLs
    (re.compile(r"\b[A-Z]{2,6}\d{6,}\b"), ""),             # BSE/NSE codes
    (re.compile(r"CIN[:\s]+[A-Z0-9]+", re.IGNORECASE), ""),
    (re.compile(r"(?:Tel|Fax|Email|Website)[:\s].{0,60}", re.IGNORECASE), ""),
    (re.compile(r"[^\x00-\x7F\u0900-\u097F]"), " "),       # keep ASCII + Devanagari
]


def remove_noise(text: str) -> str:
    for pattern, replacement in NOISE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ─────────────────────────────────────────────
# Whitespace normalisation
# ─────────────────────────────────────────────
def normalize_whitespace(text: str) -> str:
    # collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    # collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # strip trailing spaces per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


# ─────────────────────────────────────────────
# Number normalisation (financial tables)
# ─────────────────────────────────────────────
def normalize_numbers(text: str) -> str:
    # "1,23,456" → "123456"  (Indian numbering system, keep for LLM context)
    # actually just normalize crore/lakh abbreviations
    text = re.sub(r"\bCr\.?\b", "Crore", text, flags=re.IGNORECASE)
    text = re.sub(r"\bLk\.?\b", "Lakh", text, flags=re.IGNORECASE)
    text = re.sub(r"\bMn\.?\b", "Million", text, flags=re.IGNORECASE)
    text = re.sub(r"\bBn\.?\b", "Billion", text, flags=re.IGNORECASE)
    return text


# ─────────────────────────────────────────────
# Main cleaner
# ─────────────────────────────────────────────
def clean_text(text: str, aggressive: bool = False) -> str:
    """
    Clean extracted PDF text.
    aggressive=True removes more noise (good for prose, not tables).
    """
    if not text:
        return ""

    text = fix_ligatures(text)
    text = unicodedata.normalize("NFKC", text)

    if aggressive:
        text = remove_noise(text)

    text = normalize_numbers(text)
    text = normalize_whitespace(text)

    return text


def is_garbage_text(text: str, min_words: int = 10) -> bool:
    """Return True if text is too short or too noisy to be useful."""
    words = text.split()
    if len(words) < min_words:
        return True

    # ratio of alpha chars — garbled PDF text has many symbols
    alpha = sum(c.isalpha() for c in text)
    total = max(len(text), 1)
    if alpha / total < 0.4:
        return True

    return False