"""Rule-based SAE -> AAE transducer.

Covers a small set of well-documented morphosyntactic features:
  (i)   habitual 'be'
  (ii)  copula deletion
  (iii) negative concord
  (iv)  ain't  (negation of be/have)
  (v)   th-stopping in selected lexical items

The transducer is deliberately conservative: when in doubt it leaves the
input unchanged.  Content preservation is validated separately with
bidirectional NLI in ``corpus.generator``.
"""
from __future__ import annotations

import re

# (v) th-stopping is restricted to a closed set of high-frequency lexical
# lexical items where it is well attested in the linguistics literature.
# A wholesale th->d substitution would over-trigger.
_TH_STOP_TABLE = {
    r"\bthat\b": "dat",
    r"\bthese\b": "dese",
    r"\bthose\b": "dose",
    r"\bthem\b": "dem",
    r"\bthere\b": "dere",
    r"\bthen\b": "den",
    r"\bthough\b": "doh",
    r"\bthing\b": "thang",
    r"\bnothing\b": "nuthin",
    r"\bsomething\b": "sumthin",
}

# (ii) copula deletion: drop 'is/are' in predicative position.
_COPULA_DELETION = (
    (re.compile(r"\b(he|she|it|that|this)\s+is\s+", re.IGNORECASE),
     lambda m: m.group(1) + " "),
    (re.compile(r"\b(we|they|you)\s+are\s+", re.IGNORECASE),
     lambda m: m.group(1) + " "),
    # ``X is going`` and similar -- delete is before -ing.
    (re.compile(r"\b(\w+)\s+is\s+(?=\w+ing\b)", re.IGNORECASE),
     lambda m: m.group(1) + " "),
)

# (iv) ain't substitutions.  Order matters; do contractions first.
_AINT_RULES = (
    (re.compile(r"\bis not\b", re.IGNORECASE), "ain't"),
    (re.compile(r"\bare not\b", re.IGNORECASE), "ain't"),
    (re.compile(r"\bhas not\b", re.IGNORECASE), "ain't"),
    (re.compile(r"\bhave not\b", re.IGNORECASE), "ain't"),
    (re.compile(r"\bisn't\b", re.IGNORECASE), "ain't"),
    (re.compile(r"\baren't\b", re.IGNORECASE), "ain't"),
    (re.compile(r"\bhasn't\b", re.IGNORECASE), "ain't"),
    (re.compile(r"\bhaven't\b", re.IGNORECASE), "ain't"),
    (re.compile(r"\bam not\b", re.IGNORECASE), "ain't"),
    (re.compile(r"\bI'm not\b"), "I ain't"),
)

# (i) habitual 'be': map 'always is' / 'usually is' / 'is always' to bare 'be'.
_HABITUAL_BE = (
    (re.compile(r"\bis always\b", re.IGNORECASE), "be always"),
    (re.compile(r"\balways is\b", re.IGNORECASE), "always be"),
    (re.compile(r"\busually is\b", re.IGNORECASE), "usually be"),
    (re.compile(r"\bis usually\b", re.IGNORECASE), "be usually"),
)

# (iii) negative concord -- 'any/anything/anyone' under negation -> 'no/nothing/no one'.
# Realised by post-processing after ain't substitution.
_NEG_CONCORD = (
    (re.compile(r"\bain't\s+(\w+\s+)?anything\b", re.IGNORECASE),
     lambda m: "ain't " + (m.group(1) or "") + "nothin"),
    (re.compile(r"\bain't\s+(\w+\s+)?anyone\b", re.IGNORECASE),
     lambda m: "ain't " + (m.group(1) or "") + "nobody"),
    (re.compile(r"\bain't\s+(\w+\s+)?anybody\b", re.IGNORECASE),
     lambda m: "ain't " + (m.group(1) or "") + "nobody"),
    (re.compile(r"\bain't\s+got\s+any\b", re.IGNORECASE), "ain't got no"),
    (re.compile(r"\bdon't\s+(\w+\s+)?anything\b", re.IGNORECASE),
     lambda m: "don't " + (m.group(1) or "") + "nothin"),
    (re.compile(r"\bdon't\s+have\s+any\b", re.IGNORECASE), "don't have no"),
    (re.compile(r"\bdidn't\s+(\w+\s+)?anything\b", re.IGNORECASE),
     lambda m: "didn't " + (m.group(1) or "") + "nothin"),
)


def to_aae(text: str) -> str:
    """Apply the rule cascade to a civilian utterance.

    Officer utterances should NOT be passed through this function; the AAE
    rendering is restricted to non-officer speech.
    """
    out = text

    # 1. Habitual be first (it inspects 'is').
    for pat, repl in _HABITUAL_BE:
        out = pat.sub(repl, out)

    # 2. ain't substitutions.
    for pat, repl in _AINT_RULES:
        out = pat.sub(repl, out)

    # 3. Negative concord.
    for pat, repl in _NEG_CONCORD:
        out = pat.sub(repl, out)

    # 4. Copula deletion.  Skip if already removed by other rules.
    for pat, repl in _COPULA_DELETION:
        out = pat.sub(repl, out)

    # 5. th-stopping on the closed set.
    for pat, repl in _TH_STOP_TABLE.items():
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)

    # Collapse double spaces produced by copula deletion.
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out
