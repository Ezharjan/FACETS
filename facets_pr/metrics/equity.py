"""Axis E -- Equity / Dialect Parity."""
from __future__ import annotations

from typing import Mapping


def dps(metric_sae: float, metric_aae: float) -> float:
    """E1. Dialect Parity Score = M(R_AAE) - M(R_SAE).

    A model is dialect-fair on M iff ``dps`` is statistically
    indistinguishable from zero across the corpus.
    """
    if metric_sae is None or metric_aae is None:
        return float("nan")
    return float(metric_aae) - float(metric_sae)


def dps_axis_s(metrics_sae: Mapping[str, float],
               metrics_aae: Mapping[str, float]) -> dict[str, float]:
    """E2. Suspicion Drift across Dialect (SDxD).

    Returns the per-S-metric dialect drift between paired SAE / AAE runs.
    """
    out: dict[str, float] = {}
    for k in ("IIR", "PCES", "EMD", "BSI"):
        if k in metrics_sae and k in metrics_aae:
            out[k] = dps(metrics_sae[k], metrics_aae[k])
    return out
