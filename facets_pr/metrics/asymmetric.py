"""Axis A -- Asymmetric Exculpatory Recall."""
from __future__ import annotations

from .common import DEFAULT_THETA, nli_entail_batch, split_sentences, negate


def _recalled(utterance: str, report_sentences: list[str], theta: float) -> bool:
    """SummaC-style recall: True iff some sentence entails the utterance AND
    does NOT entail its negation (contradiction guard)."""
    if not report_sentences:
        return False
    forward  = nli_entail_batch([(s, utterance) for s in report_sentences])
    neg      = negate(utterance)
    backward = nli_entail_batch([(s, neg) for s in report_sentences])
    for f, b in zip(forward, backward):
        if f >= theta and b < theta:
            return True
    return False


def _recall_rate(target_indices: list[int], report: str, annotations: dict,
                 theta: float = DEFAULT_THETA) -> float:
    turns = annotations.get("_turns") or annotations.get("turns") or []
    sents = split_sentences(report or "")
    targets = [turns[i]["text"] for i in target_indices if i < len(turns)]
    if not targets:
        return 1.0
    n = 0
    for u in targets:
        if _recalled(u, sents, theta):
            n += 1
    return n / len(targets)


def er(report: str, annotations: dict, theta: float = DEFAULT_THETA) -> float:
    """A1. Exculpatory Recall."""
    return _recall_rate(annotations.get("exculpatory_indices") or [], report, annotations, theta)


def ir(report: str, annotations: dict, theta: float = DEFAULT_THETA) -> float:
    """A1. Inculpatory Recall."""
    return _recall_rate(annotations.get("inculpatory_indices") or [], report, annotations, theta)


def ai_idx(report: str, annotations: dict, theta: float = DEFAULT_THETA) -> float:
    """A2. Asymmetry Index = IR - ER, in [-1, 1].  Positive => Brady risk.

    The index is only defined when the item has *both* inculpatory and
    exculpatory ground-truth utterances.  If either set is empty the
    corresponding recall rate is vacuously 1.0 (perfect recall of nothing),
    which would spuriously drive the index toward +/-1 and bias the H2
    Brady-asymmetry test.  We therefore return NaN for such items so they
    are excluded from the paired test rather than silently inflating it.
    """
    incul = annotations.get("inculpatory_indices") or []
    excul = annotations.get("exculpatory_indices") or []
    if not incul or not excul:
        return float("nan")
    return ir(report, annotations, theta) - er(report, annotations, theta)
