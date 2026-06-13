#!/usr/bin/env python
"""Run the full FACETS-PR evaluation across all models in BWCSyn-90.

Usage:
    python run_evaluation.py \
        --corpus data/bwcsyn90.json \
        --out    results/facets_pr_results.csv \
        --models llama3.2:3b llama3.1:8b mistral:7b qwen2.5:7b phi3:mini gemma2:9b

The pipeline will pull each Ollama model if it is not already present.
"""
from __future__ import annotations

import argparse
import os
import sys

from facets_pr.pipeline import RunConfig, run, DEFAULT_MODELS


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus",  default="data/bwcsyn90.json")
    p.add_argument("--out",     default="results/facets_pr_results.csv")
    p.add_argument("--models",  nargs="*", default=list(DEFAULT_MODELS),
                   help="Ollama model tags (default: all six bundled models)")
    p.add_argument("--do-recall", action="store_true",
                   help="also compute Axis C / C2 (LRS); slower")
    p.add_argument("--no-pull", action="store_true",
                   help="skip the model-presence check (faster start-up)")
    p.add_argument("--keep-alive", default="30m",
                   help="Ollama keep_alive hint (default: 30m)")
    p.add_argument("--write-drafts-to", default=None,
                   help="optional JSONL path to dump each generated draft")
    p.add_argument("--resume", action="store_true",
                   help="skip (model, item, variant) rows already in --out CSV")
    args = p.parse_args(argv)

    if not os.path.exists(args.corpus):
        if args.resume:
            print(f"[run] ERROR: --resume requires an existing corpus at "
                  f"{args.corpus}; rebuilding would change the data.",
                  file=sys.stderr)
            return 1
        print(f"[run] corpus not found at {args.corpus}; building it now ...")
        from facets_pr.corpus.generator import build_corpus, save_corpus
        save_corpus(build_corpus(), args.corpus)

    cfg = RunConfig(
        corpus_path     = args.corpus,
        out_csv         = args.out,
        models          = tuple(args.models),
        do_recall       = args.do_recall,
        ensure_pull     = not args.no_pull,
        keep_alive      = args.keep_alive,
        write_drafts_to = args.write_drafts_to,
        resume          = args.resume,
    )
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
