"""Result figures.

Figures are written as vector PDFs with a tight bounding box so they stay
crisp at any zoom level and carry little surrounding whitespace.
"""
from __future__ import annotations

import os
from typing import Iterable, Mapping

import matplotlib
matplotlib.use("Agg")               # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import per_model_summary, per_axis_summary, METRIC_COLUMNS

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 130,
    "pdf.fonttype": 42,             # editable PDF text
})


_SAVE_KWARGS = dict(format="pdf", bbox_inches="tight", pad_inches=0.0)


def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pdf")
    fig.savefig(path, **_SAVE_KWARGS)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 1.  Per-model metric heatmap (Axis F/A/T/S only; LRS/RHDR/DPS later).
# ---------------------------------------------------------------------------

def fig_heatmap_metrics(df: pd.DataFrame, out_dir: str,
                        name: str = "fig_heatmap_metrics") -> str:
    summary = per_axis_summary(df).set_index("model")
    data = summary.values
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(summary) + 1.6))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(summary.columns)))
    ax.set_xticklabels(summary.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(summary.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(v) > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    ax.set_title("FACETS-PR metrics (SAE) per model")
    return _save(fig, out_dir, name)


# ---------------------------------------------------------------------------
# Figure 2.  FRS bar chart with bootstrap CIs.
# ---------------------------------------------------------------------------

def fig_frs_bar(df: pd.DataFrame, out_dir: str,
                name: str = "fig_frs_bar") -> str:
    from .analysis import bootstrap_ci
    sub = df[df.variant == "sae"]
    rows = []
    for m, g in sub.groupby("model"):
        lo, hi = bootstrap_ci(g["FRS"].values)
        rows.append({"model": m, "mean": float(g["FRS"].mean()), "lo": lo, "hi": hi})
    res = pd.DataFrame(rows).sort_values("mean")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    xs = np.arange(len(res))
    ax.bar(xs, res["mean"], color="#5B9BD5", edgecolor="#1F4E79")
    err_lo = res["mean"] - res["lo"]
    err_hi = res["hi"]  - res["mean"]
    ax.errorbar(xs, res["mean"], yerr=[err_lo, err_hi], fmt="none",
                color="black", capsize=4, lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(res["model"], rotation=30, ha="right")
    ax.set_ylabel("FACETS Risk Score (lower = better)")
    ax.set_title("Composite FACETS-PR Risk Score by model")
    ax.set_ylim(0, max(0.05, float(res["hi"].max()) * 1.15))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return _save(fig, out_dir, name)


# ---------------------------------------------------------------------------
# Figure 3.  Asymmetry Index distribution.
# ---------------------------------------------------------------------------

def fig_asymmetry(df: pd.DataFrame, out_dir: str,
                  name: str = "fig_asymmetry") -> str:
    sub = df[df.variant == "sae"]
    models = sorted(sub["model"].unique())
    data = [sub[sub.model == m]["AI"].dropna().values for m in models]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    # Set tick labels explicitly rather than via boxplot's ``labels`` kwarg,
    # which was renamed to ``tick_labels`` in matplotlib 3.9 and would warn
    # (and is slated for removal).
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels(models)
    ax.axhline(0, color="red", lw=1, linestyle="--")
    ax.set_ylabel("Asymmetry Index   (IR - ER)")
    ax.set_title("Brady-asymmetry of AI-assisted reports (H2)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return _save(fig, out_dir, name)


# ---------------------------------------------------------------------------
# Figure 4.  Epistemic-Modality Drift across models.
# ---------------------------------------------------------------------------

def fig_emd(df: pd.DataFrame, out_dir: str,
            name: str = "fig_emd_distribution") -> str:
    sub = df[df.variant == "sae"]
    models = sorted(sub["model"].unique())
    data = [sub[sub.model == m]["EMD"].dropna().values for m in models]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.violinplot(data, showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.axhline(0, color="red", lw=1, linestyle="--")
    ax.set_ylabel("EMD  (positive => hardened epistemic stance)")
    ax.set_title("Generative-suspicion signal (H3)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return _save(fig, out_dir, name)


# ---------------------------------------------------------------------------
# Figure 5.  Dialect Parity (DPS) heatmap, AAE rows only.
# ---------------------------------------------------------------------------

def fig_dps_heatmap(df: pd.DataFrame, out_dir: str,
                    name: str = "fig_dps_heatmap") -> str:
    dps_cols = [c for c in df.columns if c.startswith("DPS_")]
    if not dps_cols:
        return ""
    aae = df[df.variant == "aae"]
    agg = aae.groupby("model")[dps_cols].mean()
    data = agg.values
    vmax = float(np.nanmax(np.abs(data))) if data.size else 1.0
    fig, ax = plt.subplots(figsize=(9.2, 0.55 * len(agg) + 1.6))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(dps_cols)))
    ax.set_xticklabels([c.replace("DPS_", "") for c in dps_cols],
                       rotation=45, ha="right")
    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels(agg.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                    color="black" if abs(v) / max(vmax, 1e-9) < 0.6 else "white")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    ax.set_title("Dialect Parity Score  (AAE - SAE) per metric")
    return _save(fig, out_dir, name)


# ---------------------------------------------------------------------------
# Figure 6.  RHDR (Axis C) by model.
# ---------------------------------------------------------------------------

def fig_rhdr(df: pd.DataFrame, out_dir: str,
             name: str = "fig_rhdr_by_model") -> str:
    sub = df[df.variant == "sae_rh"]
    if sub.empty:
        return ""
    rates = sub.groupby("model")["RHDR"].mean()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    xs = np.arange(len(rates))
    ax.bar(xs, rates.values, color="#ED7D31", edgecolor="#C25608")
    ax.set_xticks(xs); ax.set_xticklabels(rates.index, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Red-Herring Detection Rate")
    ax.set_title("Automation-bias probe (H5):  RHDR by model")
    ax.axhline(1.0, color="green", lw=1, linestyle=":")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for x, v in zip(xs, rates.values):
        ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    return _save(fig, out_dir, name)


# ---------------------------------------------------------------------------
# Figure 7.  Per-difficulty stratified FRS.
# ---------------------------------------------------------------------------

def fig_difficulty_strata(df: pd.DataFrame, out_dir: str,
                          name: str = "fig_difficulty_strata") -> str:
    sub = df[df.variant == "sae"]
    if sub.empty or "condition" not in sub.columns:
        return ""
    pivot = sub.groupby(["condition", "model"])["FRS"].mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot.plot(kind="bar", ax=ax, edgecolor="black", width=0.85)
    ax.set_ylabel("FRS (lower better)")
    ax.set_title("FACETS Risk Score by difficulty stratum")
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best", ncol=2, fontsize=7)
    return _save(fig, out_dir, name)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def render_all(df: pd.DataFrame, out_dir: str) -> list[str]:
    out_dir = os.path.abspath(out_dir)
    paths = []
    paths.append(fig_heatmap_metrics(df, out_dir))
    paths.append(fig_frs_bar(df, out_dir))
    paths.append(fig_asymmetry(df, out_dir))
    paths.append(fig_emd(df, out_dir))
    p = fig_dps_heatmap(df, out_dir);    paths.append(p) if p else None
    p = fig_rhdr(df, out_dir);           paths.append(p) if p else None
    p = fig_difficulty_strata(df, out_dir); paths.append(p) if p else None
    return [p for p in paths if p]
