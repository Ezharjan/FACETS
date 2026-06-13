"""FACETS-PR: a reproducible framework for evaluating AI-assisted police
reports.

The package implements six evaluation axes (Fidelity, Asymmetry, Cognitive
disengagement, Equity, Traceability, Suspicion) and a composite risk score,
together with the corpus builder and statistical analysis used to drive them.
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "prompts",
    "lexicons",
    "corpus",
    "metrics",
    "pipeline",
    "analysis",
    "viz",
]
