"""Curated word lists referenced by the FACETS-PR metrics.

These lists are meant to be adapted to a deploying jurisdiction; the values
below are reasonable defaults that work for the bundled corpus.
"""
from __future__ import annotations

# Axis S (S1): legal-conclusion triggers.
LEGAL_TRIGGERS: tuple[str, ...] = (
    "probable cause",
    "reasonable suspicion",
    "intent to",
    "in furtherance of",
    "under the influence",
    "in possession of",
    "identified the suspect",
    "fled the scene",
    "refused to comply",
    "established probable cause",
)

# Axis S -- probable-cause triggers (subset of legal triggers used by PCES).
PROBABLE_CAUSE_TRIGGERS: tuple[str, ...] = (
    "probable cause",
    "reasonable suspicion",
    "established probable cause",
    "identified the suspect",
)

# Axis S (S2, PCES): sensory verbs. A probable-cause sentence that contains a
# concrete observation (a sensory verb, a time, a number, a name) is treated as
# grounded rather than as an embellishment.
SENSORY_VERBS: tuple[str, ...] = ("saw", "heard", "smelled", "observed", "watched", "noticed")

# Axis S (S3, EMD): low-certainty (hedging) modal markers.
LOW_CERTAINTY_MARKERS: tuple[str, ...] = (
    "appeared",
    "seemed",
    "may have",
    "possibly",
    "allegedly",
    "reportedly",
    "i think",
    "believe",
    "suggests",
    "might",
    "perhaps",
)

# Axis S (S3, EMD): high-certainty markers.
HIGH_CERTAINTY_MARKERS: tuple[str, ...] = (
    "clearly",
    "evidently",
    "plainly",
    "definitely",
    "must",
    "will",
    "did",
    "certainly",
    "undoubtedly",
)

# Axis S (S4, BSI): common policing boilerplate phrases.
BOILERPLATE_NGRAMS: tuple[str, ...] = (
    "in the course and scope of my duties",
    "based on my training and experience",
    "out of an abundance of caution",
    "conducted a routine patrol",
    "for officer safety",
    "in fear for my life",
    "consistent with my training",
    "in plain view",
    "freely and voluntarily",
    "without incident",
)

# Axis T (T1): the statutory disclosure string mandated by California Penal
# Code 13663(a)(1). The metric matches case-insensitively and tolerates minor
# punctuation variations.
DISCLOSURE_STRING_CANONICAL: str = (
    "This report was written either fully or in part using artificial intelligence."
)

DISCLOSURE_STRING_VARIANTS: tuple[str, ...] = (
    "this report was written either fully or in part using artificial intelligence",
    "this report was written fully or in part using artificial intelligence",
    "this report was written either fully or in part using ai",
)
