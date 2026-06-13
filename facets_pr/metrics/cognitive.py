"""Axis C -- Cognitive Disengagement."""
from __future__ import annotations

import re
from typing import Optional

import Levenshtein

from .common import normalize, split_sentences


def rhdr(reviewed_report: Optional[str], annotations: dict, tau: float = 0.30) -> Optional[float]:
    """C1. Red-Herring Detection Rate.

    Returns 1.0 if the red herring is absent from the reviewed report, 0.0 if
    still present, or ``None`` if the variant carries no red herring.

    A red herring is judged 'still present' if any sentence in the reviewed
    report matches it under a relaxed edit-distance tolerance (default 30%
    -- looser than VQR's 10%, because the officer-reviewer LLM may rephrase
    while leaving the substance untouched, which still counts as failing to
    catch the impossible content).
    """
    rh_text = annotations.get("red_herring_text")
    if rh_text is None or reviewed_report is None:
        return None
    rh_norm = normalize(rh_text)
    if not rh_norm:
        return None
    sents = split_sentences(reviewed_report)
    for s in sents:
        s_norm = normalize(s)
        # Exact substring catch.
        if rh_norm in s_norm:
            return 0.0
        # Approximate match.
        d = Levenshtein.distance(s_norm, rh_norm)
        if d / max(1, len(rh_norm)) <= tau:
            return 0.0
        # Token-overlap check: catch paraphrases that share most content words.
        rh_tokens = set(re.findall(r"\w+", rh_norm))
        s_tokens  = set(re.findall(r"\w+", s_norm))
        if rh_tokens and len(rh_tokens & s_tokens) / len(rh_tokens) >= 0.7:
            return 0.0
    return 1.0


def lrs(recalled_facts_text: str, ground_truth_facts: list[str], theta: float = 0.6) -> float:
    """C2. Latent-Recall Score.

    ``recalled_facts_text`` is the output of ``PROMPT_RECALL`` issued to a
    fresh model instance with no transcript access.  ``ground_truth_facts``
    are the proposition strings from the inculpatory + exculpatory sets.

    A ground-truth fact is considered recalled if its propositional content
    appears in the recall output under bidirectional NLI entailment.
    """
    # Avoid importing torch unless we actually need NLI (LRS may be optional).
    from .common import nli_entail_batch
    recall_sents = split_sentences(recalled_facts_text or "")
    if not ground_truth_facts:
        return 1.0
    if not recall_sents:
        return 0.0
    hits = 0
    for fact in ground_truth_facts:
        # Bidirectional entailment: a fact counts as recalled only if some
        # recall sentence both entails the fact AND is entailed by it (the min
        # of the two directions must clear theta). This guards against loose
        # one-directional matches.
        forward  = nli_entail_batch([(s, fact) for s in recall_sents])
        backward = nli_entail_batch([(fact, s) for s in recall_sents])
        if any(min(f, b) >= theta for f, b in zip(forward, backward)):
            hits += 1
    return hits / len(ground_truth_facts)
