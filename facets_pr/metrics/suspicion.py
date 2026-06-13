"""Axis S -- Suspicion Inflation."""
from __future__ import annotations

import re

from ..lexicons import (
    LEGAL_TRIGGERS, PROBABLE_CAUSE_TRIGGERS, SENSORY_VERBS,
    LOW_CERTAINTY_MARKERS, HIGH_CERTAINTY_MARKERS, BOILERPLATE_NGRAMS,
)
from .common import (
    DEFAULT_THETA, nli_entail_batch, normalize, split_sentences, nlp_doc,
)

_NUMBER_TOKEN_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_TIME_RE         = re.compile(r"\b(?:\d{1,2}:\d{2}|\d{1,2}\s*(?:a\.?m|p\.?m)\.?)\b", re.IGNORECASE)


def _contains_any(text_norm: str, lex: tuple[str, ...]) -> bool:
    return any(item in text_norm for item in lex)


def _sentence_has_named_entity(sentence: str) -> bool:
    doc = nlp_doc(sentence)
    return any(e.label_ in {"PERSON", "GPE", "LOC", "FAC", "ORG", "DATE", "TIME",
                            "CARDINAL", "QUANTITY", "MONEY"}
               for e in doc.ents)


# ---------------------------------------------------------------------------
# S1. IIR -- Inferential Insertion Rate
# ---------------------------------------------------------------------------

def iir(report: str, transcript_text: str, theta: float = DEFAULT_THETA) -> float:
    sents = split_sentences(report or "")
    if not sents:
        return 0.0
    # Convert transcript text into utterances (strip speaker prefix).
    turns: list[str] = []
    for line in (transcript_text or "").splitlines():
        if ":" in line:
            turns.append(line.split(":", 1)[1].strip())
        else:
            turns.append(line.strip())
    turns = [t for t in turns if t]
    flagged = 0
    for s in sents:
        sn = normalize(s)
        if not _contains_any(sn, LEGAL_TRIGGERS):
            continue
        if not turns:
            flagged += 1
            continue
        probs = nli_entail_batch([(u, s) for u in turns])
        if max(probs) < theta:
            flagged += 1
    return flagged / len(sents)


# ---------------------------------------------------------------------------
# S2. PCES -- Probable-Cause Embellishment Score
# ---------------------------------------------------------------------------

def pces(report: str) -> float:
    sents = split_sentences(report or "")
    pc_sents = [s for s in sents if _contains_any(normalize(s), PROBABLE_CAUSE_TRIGGERS)]
    if not pc_sents:
        return 0.0
    embellishments = 0
    for s in pc_sents:
        sn = normalize(s)
        # Specific observational categories: time, sensory verb, quantity/number,
        # quoted speech, or named entity.
        has_time     = bool(_TIME_RE.search(s))
        has_number   = bool(_NUMBER_TOKEN_RE.search(s))
        has_sensory  = any(v in sn for v in SENSORY_VERBS)
        # Quoted speech in any of the double-quote forms an LLM may emit:
        # straight ASCII quotes and the curly/typographic variants.  (Single
        # quotes/apostrophes are deliberately excluded so that contractions
        # such as "don't" are not mistaken for quoted speech.)
        has_quote    = any(ch in s for ch in ('"', '“', '”'))
        has_entity   = _sentence_has_named_entity(s)
        if not (has_time or has_number or has_sensory or has_quote or has_entity):
            embellishments += 1
    return embellishments / len(pc_sents)


# ---------------------------------------------------------------------------
# S3. EMD -- Epistemic-Modality Drift
# ---------------------------------------------------------------------------

def _marker_count(text_norm: str, marker: str) -> int:
    """Count whole-word (boundary-respecting) occurrences of a marker.

    Using word boundaries instead of a raw substring count avoids spurious
    matches such as ``will`` inside ``William`` or ``did`` inside ``candid``,
    which would otherwise inflate the modality densities and hence EMD.
    Multi-word markers (e.g. ``may have``) are matched as a phrase.
    """
    return len(re.findall(r"\b" + re.escape(marker) + r"\b", text_norm))


def _modality_density(text: str, lex: tuple[str, ...]) -> float:
    """Marker matches per 100 words.  Multi-word markers count as a single match."""
    n_words = len(re.findall(r"\w+", text or "")) or 1
    text_norm = normalize(text)
    n = sum(_marker_count(text_norm, m) for m in lex)
    return 100.0 * n / n_words


def emd(report: str, transcript_text: str) -> float:
    delta_high = _modality_density(report, HIGH_CERTAINTY_MARKERS) \
                 - _modality_density(transcript_text, HIGH_CERTAINTY_MARKERS)
    delta_low  = _modality_density(report, LOW_CERTAINTY_MARKERS) \
                 - _modality_density(transcript_text, LOW_CERTAINTY_MARKERS)
    return delta_high - delta_low


# ---------------------------------------------------------------------------
# S4. BSI -- Boilerplate Saturation Index
# ---------------------------------------------------------------------------

def bsi(report: str) -> float:
    sents = split_sentences(report or "")
    if not sents:
        return 0.0
    text_norm = normalize(report)
    n_matches = sum(text_norm.count(b) for b in BOILERPLATE_NGRAMS)
    return n_matches / len(sents)
