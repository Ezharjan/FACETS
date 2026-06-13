"""Axis F -- Source Fidelity."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

import Levenshtein

from .common import nlp_doc, normalize, cosine_align, split_sentences

_ENT_LABELS_OF_INTEREST = {"PERSON", "GPE", "LOC", "FAC", "ORG", "DATE", "TIME",
                           "PRODUCT", "EVENT", "CARDINAL", "QUANTITY", "MONEY", "WORK_OF_ART"}

_PRONOUN_GROUPS = {
    "he":  "he",  "him":  "he", "his":  "he", "himself": "he",
    "she": "she", "her":  "she", "hers": "she", "herself": "she",
    "they": "they", "them": "they", "their": "they", "themself": "they", "themselves": "they",
}


# ---------------------------------------------------------------------------
# Helper: entity extraction
# ---------------------------------------------------------------------------

def _entities_from_text(text: str) -> set[str]:
    doc = nlp_doc(text)
    out = set()
    for ent in doc.ents:
        if ent.label_ not in _ENT_LABELS_OF_INTEREST:
            continue
        s = normalize(ent.text)
        if len(s) < 2:
            continue
        out.add(s)
    return out


def _ground_truth_entity_set(annotations: dict) -> set[str]:
    return {normalize(e) for e in annotations.get("entities", []) if e}


# ---------------------------------------------------------------------------
# F1. NEHR -- Named-Entity Hallucination Rate
# ---------------------------------------------------------------------------

def nehr(report: str, annotations: dict, transcript_text: str = "") -> float:
    """Fraction of entities in the report not present in the transcript.

    The transcript universe is the union of the annotated entity inventory
    and any additional entities spaCy finds in the raw transcript text
    (this avoids penalising the model for surfacing genuinely-present
    entities that the annotator simply did not list).
    """
    rep_ents = _entities_from_text(report)
    if not rep_ents:
        return 0.0
    universe = _ground_truth_entity_set(annotations)
    if transcript_text:
        universe |= _entities_from_text(transcript_text)
    # Soft substring match: an entity 'John Smith' subsumes 'John'.
    def _present(e: str) -> bool:
        for u in universe:
            if e == u or e in u or u in e:
                return True
        return False
    hallucinated = [e for e in rep_ents if not _present(e)]
    return len(hallucinated) / len(rep_ents)


# ---------------------------------------------------------------------------
# F2. VQR -- Verbatim Quotation Retention
# ---------------------------------------------------------------------------

_QUOTE_RE = re.compile(r"[\"“]([^\"”]{4,})[\"”]")


def _extract_quotes(report: str) -> list[str]:
    return [m.group(1).strip() for m in _QUOTE_RE.finditer(report or "")]


def vqr(report: str, annotations: dict, tau: float = 0.10) -> float:
    """Fraction of evidentiary utterances preserved verbatim in the report."""
    evid = annotations.get("evidentiary_indices") or []
    turns = annotations.get("turns")
    # ``annotations`` from the pipeline carries 'turns' as a side-channel; allow
    # passing it via the broader item dict instead.
    if turns is None:
        turns = annotations.get("_turns", [])
    if not evid:
        return 1.0  # vacuously true
    evid_texts = [turns[i]["text"] for i in evid if i < len(turns)]
    quotes = _extract_quotes(report)
    if not evid_texts:
        return 1.0
    if not quotes:
        return 0.0
    n_preserved = 0
    for u in evid_texts:
        u_norm = normalize(u)
        if not u_norm:
            continue
        best_ratio = 1.0
        for q in quotes:
            q_norm = normalize(q)
            d = Levenshtein.distance(q_norm, u_norm)
            ratio = d / max(1, len(u_norm))
            if ratio < best_ratio:
                best_ratio = ratio
        if best_ratio <= tau:
            n_preserved += 1
    return n_preserved / len(evid_texts)


# ---------------------------------------------------------------------------
# F3. SAA -- Speaker Attribution Accuracy
# ---------------------------------------------------------------------------

_ATTRIB_VERBS = {"said", "told", "stated", "reported", "claimed", "explained",
                 "responded", "answered", "admitted", "denied", "asked",
                 "indicated", "remarked", "noted"}


def _extract_attributions(report: str) -> list[tuple[str, str]]:
    """Pull (subject, proposition) pairs from attribution clauses."""
    doc = nlp_doc(report)
    if len(doc) == 0:
        return []
    out: list[tuple[str, str]] = []
    for sent in doc.sents:
        for tok in sent:
            if tok.lemma_.lower() in _ATTRIB_VERBS and tok.pos_ == "VERB":
                # Find subject
                subj = None
                for child in tok.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subj = " ".join(t.text for t in child.subtree)
                        break
                # Find object / clausal complement
                obj_text = None
                for child in tok.children:
                    if child.dep_ in ("ccomp", "dobj", "xcomp", "oprd", "attr"):
                        obj_text = " ".join(t.text for t in child.subtree)
                        break
                if subj and obj_text:
                    out.append((subj.strip(), obj_text.strip()))
    return out


def _speaker_label(name: str, turns: list[dict]) -> str:
    """Map a free-form name in the report to the canonical speaker label of the
    turns list, by string match against speaker token / known entities."""
    n = normalize(name)
    if not n:
        return ""
    # Look for direct match against any turn's named speaker.
    for t in turns:
        if normalize(t["speaker"]) in n or n in normalize(t["speaker"]):
            return t["speaker"]
    # Heuristic: pronouns / titles.
    if "officer" in n: return "OFFICER"
    if any(k in n for k in ("suspect", "defendant", "the man", "the male")): return "SUSPECT"
    if any(k in n for k in ("complainant", "victim", "she said her")): return "COMPLAINANT"
    if "witness" in n or "neighbor" in n: return "WITNESS_1"
    if "dispatch" in n: return "DISPATCH"
    return ""


def saa(report: str, annotations: dict) -> float:
    """Fraction of attribution clauses whose subject matches the correct speaker."""
    turns = annotations.get("_turns") or annotations.get("turns") or []
    attributions = _extract_attributions(report)
    if not attributions:
        return 1.0  # vacuously correct
    utterances = [t["text"] for t in turns]
    n_correct = 0
    for subj, prop in attributions:
        idx = cosine_align(prop, utterances)
        if idx < 0:
            continue
        true_speaker = turns[idx]["speaker"]
        guessed = _speaker_label(subj, turns)
        if guessed == true_speaker:
            n_correct += 1
        else:
            # PRONOUN backoff: 'he/him' -> male-typed speaker label is ambiguous;
            # we don't attempt cross-mention coref here, count as wrong.
            pass
    return n_correct / len(attributions)


# ---------------------------------------------------------------------------
# F4. PCS -- Pronoun Consistency Score
# ---------------------------------------------------------------------------

def pcs(report: str) -> float:
    """1 - fraction of named referents whose coreferring pronouns span >1 gender."""
    doc = nlp_doc(report)

    # Collect PERSON mentions and group by normalized surface form.  Then for
    # each subsequent pronoun we attribute it to the nearest preceding PERSON.
    person_mentions: list[tuple[int, str]] = []
    pronoun_genders: dict[str, set[str]] = defaultdict(set)
    for tok in doc:
        if tok.ent_type_ == "PERSON" and tok.ent_iob_ == "B":
            ent = next((e for e in doc.ents if e.start == tok.i), None)
            if ent is None:
                continue
            key = normalize(ent.text.split()[0])  # head token
            person_mentions.append((tok.i, key))
        if tok.lower_ in _PRONOUN_GROUPS:
            # Nearest preceding PERSON.
            preceding = [(i, k) for (i, k) in person_mentions if i < tok.i]
            if not preceding:
                continue
            _, k = preceding[-1]
            pronoun_genders[k].add(_PRONOUN_GROUPS[tok.lower_])

    if not pronoun_genders:
        return 1.0
    inconsistent = sum(1 for genders in pronoun_genders.values() if len(genders) > 1)
    return 1.0 - inconsistent / len(pronoun_genders)
