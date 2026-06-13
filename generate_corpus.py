#!/usr/bin/env python
"""Build BWCSyn-90 and save it to disk.

Usage:
    python generate_corpus.py [--out PATH] [--seed INT]
                              [--validate-aae]
"""
from __future__ import annotations

import argparse
import sys

from facets_pr.corpus.generator import (
    build_corpus, save_corpus, validate_aae_paraphrases,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out",  default="data/bwcsyn90.json",
                   help="output JSON path (default: data/bwcsyn90.json)")
    p.add_argument("--seed", type=lambda s: int(s, 0), default=0xCDAA48,
                   help="deterministic seed (default: 0xCDAA48)")
    p.add_argument("--validate-aae", action="store_true",
                   help="run bidirectional NLI on the AAE renderings; "
                        "requires the DeBERTa NLI model.")
    args = p.parse_args(argv)

    corpus = build_corpus(seed=args.seed)
    save_corpus(corpus, args.out)
    print(f"[corpus] wrote {len(corpus)} bundles to {args.out}")

    if args.validate_aae:
        from facets_pr.metrics.common import nli_entail
        # Wrap to match the signature validate_aae_paraphrases expects.
        nli = lambda p, h: nli_entail(p, h)
        failures = validate_aae_paraphrases(corpus, nli)
        if failures:
            print(f"[corpus] WARNING: {len(failures)} AAE pairs failed the "
                  f"bidirectional 0.7 entailment threshold:")
            for f in failures[:10]:
                print(f"  - {f['id']} #{f['index']}: f={f['f']:.2f} "
                      f"b={f['b']:.2f}")
        else:
            print("[corpus] all AAE renderings pass bidirectional NLI >= 0.7")
    return 0


if __name__ == "__main__":
    sys.exit(main())
