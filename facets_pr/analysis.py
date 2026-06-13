"""Statistical analysis of the FACETS-PR results CSV."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.formula.api as smf
    _HAS_SM = True
except ImportError:
    _HAS_SM = False


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Cast numeric columns even when CSV preserves trailing whitespace, etc.
    num_cols = [c for c in df.columns if c not in
                ("model", "item_id", "category", "condition", "variant")]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Simple stats
# ---------------------------------------------------------------------------

def cohens_d(x: np.ndarray, y: np.ndarray | float | None = None) -> float:
    """Paired (one-sample) Cohen's d when y is None; otherwise standardised
    mean difference with pooled SD."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 2:
        return float("nan")
    if y is None or np.isscalar(y):
        baseline = 0.0 if y is None else float(y)
        sd = x.std(ddof=1)
        return (x.mean() - baseline) / sd if sd > 0 else float("nan")
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return float("nan")
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * sx**2 + (n2 - 1) * sy**2) / (n1 + n2 - 2))
    return (x.mean() - y.mean()) / pooled if pooled > 0 else float("nan")


def bootstrap_ci(x: np.ndarray, n: int = 2000, ci: float = 0.95,
                 seed: int = 0xCDAA48) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (float("nan"), float("nan"))
    means = rng.choice(x, size=(n, x.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def one_sample_t(x: np.ndarray, mu: float = 0.0,
                 alternative: str = "greater") -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 2:
        return {"t": float("nan"), "p": float("nan"), "n": int(x.size)}
    t, p = stats.ttest_1samp(x, mu, alternative=alternative)
    return {"t": float(t), "p": float(p), "n": int(x.size),
            "mean": float(x.mean()), "sd": float(x.std(ddof=1))}


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------

def test_h1_verbatim_disappearance(df: pd.DataFrame) -> pd.DataFrame:
    """H1: VQR < 0.5 for every model on average (one-sided, mu = 0.5)."""
    rows = []
    for m, g in df[df.variant == "sae"].groupby("model"):
        r = one_sample_t(g["VQR"].values, mu=0.5, alternative="less")
        r["d"] = cohens_d(g["VQR"].values, y=0.5)
        rows.append({"model": m, **r})
    out = pd.DataFrame(rows)
    out["p_bonferroni"] = (out["p"] * max(1, len(out))).clip(upper=1.0)
    return out


def test_h2_brady_asymmetry(df: pd.DataFrame) -> pd.DataFrame:
    """H2: AI > 0 across the 30 base items per model (paired, mu = 0, greater)."""
    rows = []
    for m, g in df[df.variant == "sae"].groupby("model"):
        r = one_sample_t(g["AI"].values, mu=0.0, alternative="greater")
        r["d"] = cohens_d(g["AI"].values)
        rows.append({"model": m, **r})
    out = pd.DataFrame(rows)
    out["p_bonferroni"] = (out["p"] * max(1, len(out))).clip(upper=1.0)
    return out


def test_h3_generative_suspicion(df: pd.DataFrame) -> pd.DataFrame:
    """H3: EMD > 0 and IIR > 0 for at least one model significantly."""
    rows = []
    for m, g in df[df.variant == "sae"].groupby("model"):
        r1 = one_sample_t(g["EMD"].values, mu=0.0, alternative="greater")
        r2 = one_sample_t(g["IIR"].values, mu=0.0, alternative="greater")
        rows.append({
            "model": m,
            "EMD_t": r1["t"], "EMD_p": r1["p"], "EMD_d": cohens_d(g["EMD"].values),
            "IIR_t": r2["t"], "IIR_p": r2["p"], "IIR_d": cohens_d(g["IIR"].values),
        })
    return pd.DataFrame(rows)


def test_h4_dialect_disparity(df: pd.DataFrame) -> pd.DataFrame:
    """H4: DPS_EMD > 0 on AAE-paired transcripts."""
    rows = []
    for m, g in df[df.variant == "aae"].groupby("model"):
        r = one_sample_t(g["DPS_EMD"].values, mu=0.0, alternative="greater")
        r["d"] = cohens_d(g["DPS_EMD"].values)
        rows.append({"model": m, **r})
    return pd.DataFrame(rows)


def test_h5_automation_bias(df: pd.DataFrame) -> pd.DataFrame:
    """H5: RHDR < 1 -- officer reviewers leave at least some red herrings."""
    rows = []
    sub = df[df.variant == "sae_rh"]
    for m, g in sub.groupby("model"):
        # Bernoulli test against H0: RHDR == 1 (perfect detection).
        x = g["RHDR"].dropna().values
        n = len(x)
        successes = int(np.sum(x == 0))   # times red herring slipped through
        # Two-sided exact binomial against p = 0 baseline (all caught).
        if n == 0:
            rows.append({"model": m, "n": 0, "mean_rhdr": float("nan"),
                         "missed": 0, "p": float("nan")})
            continue
        mean = float(x.mean())
        # P(any miss | true perfect) = 0; so any miss => p ~ 0.
        rows.append({
            "model":     m,
            "n":         n,
            "mean_rhdr": mean,
            "missed":    int(n - x.sum()),
            "p":         0.0 if successes > 0 else 1.0,
        })
    return pd.DataFrame(rows)


def test_h6_disclosure_non_spontaneity(df: pd.DataFrame) -> pd.DataFrame:
    """H6: DTC == 0 unless the prompt explicitly requests it."""
    rows = []
    for m, g in df.groupby("model"):
        x = g["DTC"].dropna().values
        rows.append({
            "model": m,
            "n":     int(x.size),
            "rate":  float(x.mean()) if x.size else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mixed-effects (H3 / H4 / H5 jointly)
# ---------------------------------------------------------------------------

def mixed_effects(df: pd.DataFrame, dependent: str = "EMD") -> Optional[object]:
    """Fit a linear mixed-effects model with random intercepts for (model, item).

    Specifically:
        M_{ijk} = b0 + b1 * 1[AAE] + b2 * difficulty + u_i + v_k + eps

    Requires statsmodels.  Returns the fit result or ``None`` if statsmodels
    is not installed.
    """
    if not _HAS_SM:
        warnings.warn("statsmodels not installed; skipping mixed-effects fit.")
        return None
    d = df[df.variant.isin(["sae", "aae"])].copy()
    d["is_aae"]   = (d["variant"] == "aae").astype(int)
    d["item_id"]  = d["item_id"].astype(str)
    d["model"]    = d["model"].astype(str)
    formula = f"{dependent} ~ is_aae + C(condition)"
    model = smf.mixedlm(formula, d, groups=d["model"], re_formula="~1",
                        vc_formula={"item": "0 + C(item_id)"})
    try:
        fit = model.fit(reml=True, method="lbfgs", maxiter=200)
    except Exception:
        # Fallback: drop the nested item-level variance term.
        model = smf.mixedlm(formula, d, groups=d["model"], re_formula="~1")
        fit = model.fit(reml=True, method="lbfgs", maxiter=200)
    return fit


# ---------------------------------------------------------------------------
# Aggregations for plotting
# ---------------------------------------------------------------------------

METRIC_COLUMNS = ["NEHR", "VQR", "SAA", "PCS",
                  "ER", "IR", "AI",
                  "DTC", "SCS",
                  "IIR", "PCES", "EMD", "BSI",
                  "FRS"]


def per_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df.variant == "sae"].groupby("model")[METRIC_COLUMNS].mean()
    return out.reset_index()


def per_axis_summary(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df.variant == "sae"]
    agg = {
        "F.NEHR": sub.groupby("model")["NEHR"].mean(),
        "F.VQR":  sub.groupby("model")["VQR"].mean(),
        "F.SAA":  sub.groupby("model")["SAA"].mean(),
        "F.PCS":  sub.groupby("model")["PCS"].mean(),
        "A.ER":   sub.groupby("model")["ER"].mean(),
        "A.IR":   sub.groupby("model")["IR"].mean(),
        "A.AI":   sub.groupby("model")["AI"].mean(),
        "T.DTC":  sub.groupby("model")["DTC"].mean(),
        "T.SCS":  sub.groupby("model")["SCS"].mean(),
        "S.IIR":  sub.groupby("model")["IIR"].mean(),
        "S.PCES": sub.groupby("model")["PCES"].mean(),
        "S.EMD":  sub.groupby("model")["EMD"].mean(),
        "S.BSI":  sub.groupby("model")["BSI"].mean(),
        "FRS":    sub.groupby("model")["FRS"].mean(),
    }
    return pd.DataFrame(agg).reset_index()
