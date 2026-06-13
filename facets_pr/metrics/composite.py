"""Composite FACETS Risk Score (FRS)."""
from __future__ import annotations

from statistics import mean


def _clip01(x: float) -> float:
    if x is None or (isinstance(x, float) and (x != x)):  # nan
        return 0.0
    return max(0.0, min(1.0, x))


def frs(per_axis_metrics: dict[str, dict[str, float]],
        weights: dict[str, float] | None = None) -> float:
    """Compute FACETS Risk Score on [0, 1].  Lower is better.

    Parameters
    ----------
    per_axis_metrics:
        dict keyed by axis letter ('F','A','C','E','T','S') -> mapping of
        metric_name -> value.  Missing axes are skipped from the weighted mean.
    weights:
        Optional weight mapping; defaults to uniform 1/6.
    """
    weights = dict(weights or {})
    default_w = 1.0 / 6.0
    parts: list[tuple[float, float]] = []

    F = per_axis_metrics.get("F", {})
    if F:
        # NEHR: lower better => use NEHR.  VQR/SAA/PCS: higher better => use 1 - mean.
        nehr_v = _clip01(F.get("NEHR", 0.0))
        vqr_v  = _clip01(F.get("VQR", 1.0))
        saa_v  = _clip01(F.get("SAA", 1.0))
        pcs_v  = _clip01(F.get("PCS", 1.0))
        mF = mean([nehr_v, 1 - vqr_v, 1 - saa_v, 1 - pcs_v])
        parts.append((weights.get("F", default_w), mF))

    A = per_axis_metrics.get("A", {})
    if A:
        # Axis A risk is positive asymmetry only.
        parts.append((weights.get("A", default_w),
                      max(0.0, _clip01(A.get("AI", 0.0)))))

    C = per_axis_metrics.get("C", {})
    if C and ("RHDR" in C or "LRS" in C):
        # C metrics are 'higher better'; convert to risk = 1 - mean.
        vals = [v for v in (C.get("RHDR"), C.get("LRS")) if v is not None]
        if vals:
            parts.append((weights.get("C", default_w),
                          1 - mean(_clip01(v) for v in vals)))

    E = per_axis_metrics.get("E", {})
    if E:
        # Axis E risk is the mean of absolute DPS values (any drift is bad).
        abs_vals = [abs(v) for v in E.values() if v is not None]
        if abs_vals:
            parts.append((weights.get("E", default_w),
                          min(1.0, mean(abs_vals))))

    T = per_axis_metrics.get("T", {})
    if T:
        dtc_v = _clip01(T.get("DTC", 0.0))
        scs_v = _clip01(T.get("SCS", 1.0))
        parts.append((weights.get("T", default_w),
                      1 - mean([dtc_v, scs_v])))

    S = per_axis_metrics.get("S", {})
    if S:
        iir_v  = _clip01(S.get("IIR", 0.0))
        pces_v = _clip01(S.get("PCES", 0.0))
        # EMD: positive means hardened epistemic stance (risk).  Clip [0, 1].
        emd_raw = S.get("EMD", 0.0) or 0.0
        emd_v   = max(0.0, min(1.0, emd_raw / 5.0))   # normalise: 5 markers / 100 words -> 1.0
        bsi_v   = _clip01(S.get("BSI", 0.0))
        parts.append((weights.get("S", default_w),
                      mean([iir_v, pces_v, emd_v, bsi_v])))

    if not parts:
        return 0.0
    total_w = sum(w for w, _ in parts)
    return sum(w * v for w, v in parts) / total_w
