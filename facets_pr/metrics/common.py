"""Shared utilities for FACETS-PR metrics.

GPU usage
---------
* ``spaCy`` -- ``en_core_web_trf`` runs on GPU when ``cupy`` + ``spacy[cuda]``
  are available; we attempt ``spacy.require_gpu()`` and fall back silently.
* ``transformers`` -- NLI uses CUDA when ``torch.cuda.is_available()``.
* ``sentence-transformers`` -- defaults to CUDA when available.

All loaders are memoised so that within a single Python process the heavy
models are downloaded / instantiated once.
"""
from __future__ import annotations

import functools
import re
import warnings
from typing import Callable, Optional

import torch

DEFAULT_THETA = 0.6                # SummaC-style entailment threshold (Axis A).
DEFAULT_TAU   = 0.10               # VQR edit-distance tolerance (Axis F).

# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# spaCy
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def load_spacy(model: str = "en_core_web_trf"):
    """Load a spaCy pipeline, preferring GPU."""
    import spacy
    try:
        spacy.require_gpu()
    except Exception:
        # No CUDA-capable spaCy installation; fall back silently.
        pass
    try:
        nlp = spacy.load(model)
    except OSError:
        # Transformer model not downloaded; fall back to the lemma-tagger pipeline.
        warnings.warn(
            f"spaCy model '{model}' not found.  Falling back to 'en_core_web_sm'.  "
            f"Run:  python -m spacy download {model}"
        )
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            warnings.warn(
                "Neither '%s' nor 'en_core_web_sm' is installed.  Install with "
                "'python -m spacy download en_core_web_sm'." % model
            )
            raise
    return nlp


_HAS_WORD_CHAR_RE = re.compile(r"\w")


def nlp_doc(text: str):
    """Run the full spaCy pipeline on ``text``, robust to degenerate input.

    LLMs occationally emit degenerate drafts: empty strings, whitespace-only
    text (e.g. hundreds of newlines), or text containing no word characters.
    Such inputs make the transformer pipeline produce a zero-width feature
    tensor, which crashes thinc downstream with::

        ValueError: Shape mismatch for blis.gemm: (1, 0), (768, 49)

    This wrapper (i) collapses whitespace runs, (ii) short-circuits inputs
    with no word characters to an empty tokenizer-only Doc, and (iii) falls
    back to an empty Doc if the pipeline still fails, so a single degenerate
    draft cannot abort a multi-hour evaluation run.

    Callers must therefore tolerate a Doc with zero tokens (no ``ents``;
    do not iterate ``sents`` without checking ``len(doc)`` first).
    """
    nlp = load_spacy()
    t = _WHITESPACE_RE.sub(" ", (text or "")).strip()
    if not t or not _HAS_WORD_CHAR_RE.search(t):
        return nlp.make_doc("")
    try:
        return nlp(t)
    except Exception as e:  # pragma: no cover - defensive
        warnings.warn(
            f"spaCy pipeline failed on degenerate text ({e!r}); "
            "returning an empty Doc instead."
        )
        return nlp.make_doc("")


# ---------------------------------------------------------------------------
# Sentence splitter (cheap, doesn't need spaCy to be hot)
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text)]
    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# NLI (DeBERTa-v3-MNLI)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def load_nli_model(name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
    """Load DeBERTa-v3 MNLI-fever-anli and return ``(tokenizer, model, label2id)``.

    This checkpoint exposes the canonical NLI label space.  We map class
    indices via the model config so that we are robust to checkpoint
    re-indexing.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name).to(device()).eval()
    id2label = {int(k): v.lower() for k, v in mdl.config.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    return tok, mdl, label2id


@torch.inference_mode()
def nli_entail(premise: str, hypothesis: str) -> float:
    """Probability the premise entails the hypothesis (single pair)."""
    tok, mdl, label2id = load_nli_model()
    enc = tok(premise, hypothesis, return_tensors="pt",
              truncation=True, max_length=512).to(mdl.device)
    logits = mdl(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1)
    return float(probs[label2id["entailment"]].cpu().item())


@torch.inference_mode()
def nli_entail_batch(pairs: list[tuple[str, str]], batch_size: int = 16) -> list[float]:
    """Batched ``P(entail)`` for a list of ``(premise, hypothesis)`` pairs."""
    if not pairs:
        return []
    tok, mdl, label2id = load_nli_model()
    out: list[float] = []
    entail_idx = label2id["entailment"]
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        prem = [p for p, _ in batch]
        hypo = [h for _, h in batch]
        enc = tok(prem, hypo, return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(mdl.device)
        logits = mdl(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        out.extend(probs[:, entail_idx].detach().cpu().tolist())
    return out


# ---------------------------------------------------------------------------
# Sentence embeddings (for SAA alignment via BERTScore-style cosine)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def load_sentence_encoder(name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device=device())


def encode(texts: list[str]):
    if not texts:
        return torch.empty(0, 384)
    encoder = load_sentence_encoder()
    return encoder.encode(texts, convert_to_tensor=True, show_progress_bar=False)


def cosine_align(query: str, candidates: list[str]) -> int:
    """Return the index of the candidate with maximal cosine similarity to query."""
    if not candidates:
        return -1
    q = encode([query])
    c = encode(candidates)
    # Cosine sim
    q_n = torch.nn.functional.normalize(q, dim=-1)
    c_n = torch.nn.functional.normalize(c, dim=-1)
    sims = (c_n @ q_n.T).squeeze(-1)
    return int(torch.argmax(sims).item())


# ---------------------------------------------------------------------------
# Light text helpers
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").lower()).strip()


def negate(sentence: str) -> str:
    """Naive deterministic negation used by the contradiction guard in the
    recall metrics. Inserts 'not' or removes an existing negation."""
    s = sentence.strip()
    if re.search(r"\b(not|n't|never|no)\b", s, flags=re.IGNORECASE):
        s2 = re.sub(r"\bnot\b",   "", s, flags=re.IGNORECASE)
        s2 = re.sub(r"n't\b",     "", s2)
        s2 = re.sub(r"\bnever\b", "ever", s2, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", s2).strip()
    m = re.search(r"\b(is|are|was|were|will|would|can|could|did|do|does|has|have|had)\b", s, re.IGNORECASE)
    if m:
        i, j = m.span()
        return s[:j] + " not " + s[j:]
    return "It is not the case that " + s
