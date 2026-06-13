#!/usr/bin/env python
"""Run the hypothesis tests and render the result figures.

Usage:
    python analyze_results.py \
        --csv  results/facets_pr_results.csv \
        --figs results/figures \
        --report results/report.txt
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from facets_pr.analysis import (
    load_results, test_h1_verbatim_disappearance, test_h2_brady_asymmetry,
    test_h3_generative_suspicion, test_h4_dialect_disparity,
    test_h5_automation_bias, test_h6_disclosure_non_spontaneity,
    mixed_effects, per_model_summary, per_axis_summary,
)
from facets_pr.viz import render_all


def _write_section(fh, name: str, df: pd.DataFrame) -> None:
    fh.write(f"\n=== {name} ===\n")
    fh.write(df.to_string(index=False, float_format=lambda x: f"{x:.4f}") + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv",    default="results/facets_pr_results.csv")
    p.add_argument("--figs",   default="results/figures")
    p.add_argument("--report", default="results/report.txt")
    args = p.parse_args(argv)

    if not os.path.exists(args.csv):
        print(f"[analyze] {args.csv} does not exist; run run_evaluation.py first.",
              file=sys.stderr)
        return 1

    df = load_results(args.csv)
    print(f"[analyze] loaded {len(df)} rows from {args.csv}")

    # ---- Hypothesis tests -------------------------------------------------
    h1 = test_h1_verbatim_disappearance(df)
    h2 = test_h2_brady_asymmetry(df)
    h3 = test_h3_generative_suspicion(df)
    h4 = test_h4_dialect_disparity(df)
    h5 = test_h5_automation_bias(df)
    h6 = test_h6_disclosure_non_spontaneity(df)
    pm = per_model_summary(df)
    pa = per_axis_summary(df)

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write("FACETS-PR  --  hypothesis tests\n")
        fh.write("=" * 60 + "\n")
        _write_section(fh, "H1  Verbatim Disappearance (VQR < 0.5, one-sided)", h1)
        _write_section(fh, "H2  Brady Asymmetry (AI > 0, one-sided)", h2)
        _write_section(fh, "H3  Generative Suspicion (EMD > 0; IIR > 0)", h3)
        _write_section(fh, "H4  Dialect Disparity (DPS_EMD > 0)", h4)
        _write_section(fh, "H5  Automation Bias (RHDR < 1)", h5)
        _write_section(fh, "H6  Disclosure Non-Spontaneity", h6)
        _write_section(fh, "Per-model summary (SAE)", pm)
        _write_section(fh, "Per-axis summary (SAE)", pa)

        # Mixed-effects (optional)
        for dep in ("EMD", "IIR"):
            fit = mixed_effects(df, dep)
            if fit is None:
                fh.write(f"\n[mixed-effects {dep}] statsmodels missing; skipped.\n")
                continue
            fh.write(f"\n=== Mixed-effects fit -- {dep} ===\n")
            fh.write(str(fit.summary()) + "\n")

    print(f"[analyze] wrote {args.report}")

    # ---- Figures ----------------------------------------------------------
    paths = render_all(df, args.figs)
    for p_ in paths:
        print(f"[figs]  wrote {p_}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
