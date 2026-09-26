"""Text normalization for business names and addresses.

Each optional step is a small function. `normalize(text, steps)` always lowercases and removes
symbols; the optional steps listed in `steps` are applied in the fixed order of STEP_ORDER.

Regex note: pandas 3 runs .str.replace with the RE2 engine, where \\w is ASCII-only.
We therefore use \\p{L} (letter), \\p{N} (digit), \\p{M} (mark, e.g. Hindi vowel signs).
"""

import pandas as pd

# ---------------------------------------------------------------- steps on lowercased raw text

TRADE_NAME_MARKER = r"^.+?\s(?:d\.?b\.?a\.?|a\.?k\.?a\.?|née|formerly known as|formerly|t/a|trading as):?\s+"


def keep_real_name(text):
    """'zephflux formerly: cp group' -> 'cp group'. Only when something comes BEFORE the marker,
    so genuine names such as 'dba brothers pvt ltd' are left alone."""
    return text.str.replace(TRADE_NAME_MARKER, "", regex=True)


def remove_invisible(text):
    """Delete zero-width characters (used inside Indian-script words) and soft hyphens."""
    return text.str.replace("[​-‍﻿­]", "", regex=True)


def strip_accents(text):
    """'école límited' -> 'ecole limited'. NFKD splits é into e + ´; we delete only the Latin accent marks
    (U+0300-U+036F), so Indian vowel signs are untouched."""
    return text.str.normalize("NFKD").str.replace("[̀-ͯ]", "", regex=True)


def ampersand_to_and(text):
    return text.str.replace("&", " and ", regex=False)


def drop_web(text):
    """'mejiastrategictalon.com' -> 'mejiastrategictalon', 'www.vayuresor.com' -> 'vayuresor'."""
    text = text.str.replace(r"\bwww\.", "", regex=True)
    return text.str.replace(r"\.(?:com|net|org|co\.in|in|co|fr)\b", " ", regex=True)


def join_dotted(text):
    """Delete dots instead of turning them into spaces: 'l.l.c.' -> 'llc', 'pvt.' -> 'pvt'."""
    return text.str.replace(".", "", regex=False)


# ---------------------------------------------------------------- steps on cleaned text (space-separated words)

def replace_words(text, mapping):
    """Replace whole words using a {word: replacement} dict (replacement may be '' to delete).

    Works on space-separated text: we pad with spaces and replace ' word ' literally (twice, so
    'st st' is handled). Literal replacement is much faster than regex on millions of rows.
    """
    text = " " + text + " "
    for word, replacement in mapping.items():
        new = f" {replacement} " if replacement else " "
        text = text.str.replace(f" {word} ", new, regex=False).str.replace(f" {word} ", new, regex=False)
    return text.str.replace(r"\s+", " ", regex=True).str.strip()


LEGAL_CANONICAL = {
    "pvt": "private", "pvt ltd": "private limited", "ltd": "limited",
    "inc": "incorporated", "corp": "corporation", "co": "company",
    "l l c": "llc", "l l p": "llp", "p c": "pc",
}

LEGAL_WORDS = [
    "private", "pvt", "limited", "ltd", "public", "llc", "llp", "pllc", "inc", "incorporated",
    "corp", "corporation", "co", "company", "pc", "lp", "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "ei",
]

ADDRESS_ABBREVIATIONS = {
    "st": "street", "rd": "road", "dr": "drive", "ave": "avenue", "av": "avenue", "ln": "lane",
    "blvd": "boulevard", "bd": "boulevard", "ct": "court", "cir": "circle", "pl": "place",
    "pkwy": "parkway", "hwy": "highway", "apt": "apartment", "r": "rue",
}

ADDRESS_PLACEHOLDERS = {"null": ""}


def canonical_legal(text):
    return replace_words(text, LEGAL_CANONICAL)


def remove_legal(text):
    return replace_words(text, {word: "" for word in LEGAL_WORDS})


def expand_address_abbreviations(text):
    return replace_words(text, ADDRESS_ABBREVIATIONS)


def drop_placeholders(text):
    return replace_words(text, ADDRESS_PLACEHOLDERS)


def strip_leading_zeros(text):
    """'no 045' -> 'no 45' ('0' alone stays)."""
    return text.str.replace(r"\b0+(\d)", r"\1", regex=True)


# ---------------------------------------------------------------- the pipeline

RAW_STEPS = {  # applied to lowercased raw text, before symbols are removed
    "trade_name": keep_real_name,
    "invisible": remove_invisible,
    "accents": strip_accents,
    "ampersand": ampersand_to_and,
    "web": drop_web,
    "dots": join_dotted,
}
WORD_STEPS = {  # applied after symbols are removed
    "legal_canonical": canonical_legal,
    "legal_remove": remove_legal,
    "address_abbrev": expand_address_abbreviations,
    "null": drop_placeholders,
    "zeros": strip_leading_zeros,
}
STEP_ORDER = list(RAW_STEPS) + list(WORD_STEPS)

# Chosen in Phase 6 (notebook 06): +0.010 macro F0.5 over basic cleaning, 95% CI [+0.0065, +0.0134].
# Rejected because they hurt: legal_canonical, legal_remove, dots; ampersand was neutral.
NAME_STEPS = ("trade_name", "invisible", "accents", "web")
ADDRESS_STEPS = ("invisible", "accents", "address_abbrev", "null", "zeros")


def basic_clean(text):
    """Symbols -> space, squeeze spaces. Letters, digits and marks of every script are kept."""
    return (
        text.str.replace(r"[^\p{L}\p{N}\p{M}\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def normalize(text, steps=()):
    """Lowercase + optional raw steps + symbol removal + optional word steps."""
    unknown = set(steps) - set(STEP_ORDER)
    if unknown:
        raise ValueError(f"unknown normalization steps: {unknown}")
    text = text.str.lower()
    for step in RAW_STEPS:
        if step in steps:
            text = RAW_STEPS[step](text)
    text = basic_clean(text)
    for step in WORD_STEPS:
        if step in steps:
            text = WORD_STEPS[step](text)
    return text
