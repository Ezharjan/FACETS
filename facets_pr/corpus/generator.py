"""Builds and serialises the BWCSyn-90 corpus."""
from __future__ import annotations

import hashlib
import json
import os
import random
from copy import deepcopy

from .templates import all_base_items
from .aae import to_aae
from .red_herrings import sample_red_herring

_OFFICER_LIKE = {"OFFICER", "DISPATCH"}


def _stable_hash(text: str) -> int:
    """Deterministic 32-bit hash of a string.

    Python's built-in ``hash()`` is salted per process (PYTHONHASHSEED), so
    using it here would make corpus construction non-reproducible across runs,
    which would break the fixed-seed guarantee described in the README. BLAKE2b
    is stable everywhere, so we use that instead.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def _collect_annotations(turns):
    evid, incul, excul = [], [], []
    for i, t in enumerate(turns):
        tags = t.get("tags", [])
        if "evid"  in tags: evid.append(i)
        if "incul" in tags: incul.append(i)
        if "excul" in tags: excul.append(i)
    return {
        "evidentiary_indices":  evid,
        "inculpatory_indices":  incul,
        "exculpatory_indices":  excul,
    }


def _render_transcript(turns, evidentiary_idx):
    lines = []
    evid_set = set(evidentiary_idx)
    for i, t in enumerate(turns):
        text = t["text"]
        if i in evid_set:
            text = f"<Q>{text}</Q>"
        lines.append(f"{t['speaker']}: {text}")
    return "\n".join(lines)


def _make_aae_variant(item):
    out = deepcopy(item)
    for t in out["turns"]:
        if t["speaker"] not in _OFFICER_LIKE:
            t["text"] = to_aae(t["text"])
    out["variant"] = "aae"
    return out


def _shift_indices(idxs, pos):
    return [i + 1 if i >= pos else i for i in idxs]


def _make_rh_variant(item, seed):
    out = deepcopy(item)
    rng = random.Random(seed)
    pos = rng.randint(2, max(2, len(out["turns"]) - 1))
    rh_text = sample_red_herring(seed)
    rh_turn = {"speaker": "WITNESS_RH", "text": rh_text, "tags": ["red_herring"]}
    out["turns"].insert(pos, rh_turn)

    ann = out["annotations"]
    ann["evidentiary_indices"] = _shift_indices(ann.get("evidentiary_indices", []), pos)
    ann["inculpatory_indices"] = _shift_indices(ann.get("inculpatory_indices", []), pos)
    ann["exculpatory_indices"] = _shift_indices(ann.get("exculpatory_indices", []), pos)
    ann["red_herring_text"]   = rh_text
    ann["red_herring_index"]  = pos
    out["variant"] = "sae_rh"
    return out


def build_corpus(seed=0xCDAA48):
    rng = random.Random(seed)
    corpus = []
    for base in all_base_items():
        item_seed = (seed ^ _stable_hash(base["id"])) & 0xFFFFFFFF

        sae = deepcopy(base)
        sae["annotations"] = _collect_annotations(sae["turns"])
        sae["annotations"]["entities"] = list(base["entities"])
        sae["transcript_text"] = _render_transcript(
            sae["turns"], sae["annotations"]["evidentiary_indices"]
        )
        sae["variant"] = "sae"
        sae["item_seed"] = item_seed

        aae = _make_aae_variant(sae)
        aae["transcript_text"] = _render_transcript(
            aae["turns"], aae["annotations"]["evidentiary_indices"]
        )

        rh = _make_rh_variant(sae, item_seed)
        rh["transcript_text"] = _render_transcript(
            rh["turns"], rh["annotations"]["evidentiary_indices"]
        )

        corpus.append({
            "id":         base["id"],
            "category":   base["category"],
            "condition":  base["condition"],
            "entities":   list(base["entities"]),
            "sae":        sae,
            "aae":        aae,
            "sae_rh":     rh,
        })

    rng.shuffle(corpus)
    return corpus


def save_corpus(corpus, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(corpus, fh, ensure_ascii=False, indent=2)


def load_corpus(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_aae_paraphrases(corpus, nli):
    failures = []
    for bundle in corpus:
        sae_turns = bundle["sae"]["turns"]
        aae_turns = bundle["aae"]["turns"]
        for i, (s, a) in enumerate(zip(sae_turns, aae_turns)):
            if s["speaker"] in _OFFICER_LIKE:
                continue
            if s["text"] == a["text"]:
                continue
            f = nli(s["text"], a["text"])
            b = nli(a["text"], s["text"])
            if min(f, b) < 0.7:
                failures.append({"id": bundle["id"], "index": i, "f": f, "b": b})
    return failures
