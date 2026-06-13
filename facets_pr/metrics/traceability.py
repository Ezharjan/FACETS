"""Axis T -- Traceability."""
from __future__ import annotations

from ..lexicons import DISCLOSURE_STRING_VARIANTS
from .common import DEFAULT_THETA, nli_entail_batch, normalize, split_sentences


def dtc(report: str) -> float:
    """T1. Disclosure-Tag Compliance.  1.0 if the statutory string appears in
    the report (case-insensitively, allowing minor variants), else 0.0."""
    r = normalize(report)
    for v in DISCLOSURE_STRING_VARIANTS:
        if v in r:
            return 1.0
    return 0.0


def scs(report: str, transcript_text: str, theta: float = DEFAULT_THETA) -> float:
    """T2. Source-Citability Score.  Fraction of report sentences alignable
    (NLI entail >= theta) to at least one transcript turn."""
    sents = split_sentences(report or "")
    if not sents:
        return 1.0
    # Convert transcript lines into clean turns by stripping speaker prefix.
    turns: list[str] = []
    for line in (transcript_text or "").splitlines():
        if ":" in line:
            turns.append(line.split(":", 1)[1].strip())
        else:
            turns.append(line.strip())
    turns = [t for t in turns if t]
    if not turns:
        return 0.0
    citable = 0
    for s in sents:
        pairs = [(u, s) for u in turns]
        probs = nli_entail_batch(pairs)
        if probs and max(probs) >= theta:
            citable += 1
    return citable / len(sents)
