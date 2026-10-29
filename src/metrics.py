"""Risk metrics: KS, PR-AUC, PSI, calibration, capture rate.

Accuracy is absent on purpose. At a 13% bad rate, predicting 'everyone is good'
scores 87% accurate and approves every defaulter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Max gap between the cumulative bad and good distributions.

    Rule of thumb in retail credit: <0.20 weak, 0.20-0.40 usable,
    >0.50 suspicious — check for leakage rather than claiming a win.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(y_score)
    y_sorted = y_true[order]

    bads, goods = y_sorted.sum(), len(y_sorted) - y_sorted.sum()
    if bads == 0 or goods == 0:
        return float("nan")

    cum_bad = np.cumsum(y_sorted) / bads
    cum_good = np.cumsum(1 - y_sorted) / goods
    return float(np.max(np.abs(cum_bad - cum_good)))


def capture_at_top(y_true: np.ndarray, y_score: np.ndarray, frac: float = 0.10) -> float:
    """Share of all bads sitting in the riskiest `frac` of the population."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n = max(1, int(round(len(y_score) * frac)))
    top = np.argsort(-y_score)[:n]
    total_bads = y_true.sum()
    return float(y_true[top].sum() / total_bads) if total_bads else float("nan")


def summary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    return {
        "n": int(len(y_true)),
        "bad_rate": round(float(y_true.mean()), 4),
        "auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "ks": round(ks_statistic(y_true, y_score), 4),
        "capture_top_10pct": round(capture_at_top(y_true, y_score, 0.10), 4),
        "capture_top_20pct": round(capture_at_top(y_true, y_score, 0.20), 4),
    }


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between a baseline and a later sample.

    Bin edges come from `expected` (the training distribution) — using each
    sample's own quantiles would hide the very shift PSI exists to detect.
    Convention: <0.10 stable, 0.10-0.25 watch, >0.25 investigate / retrain.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    edges = np.unique(np.nanquantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    e_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    a_pct = np.histogram(actual, bins=edges)[0] / len(actual)

    eps = 1e-6
    e_pct, a_pct = np.clip(e_pct, eps, None), np.clip(a_pct, eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_table(baseline: pd.Series, periods: dict[str, pd.Series],
              bins: int = 10) -> pd.DataFrame:
    rows = []
    for name, sample in periods.items():
        value = psi(baseline.to_numpy(), sample.to_numpy(), bins=bins)
        rows.append({
            "period": name,
            "n": int(sample.notna().sum()),
            "psi": round(value, 4),
            "verdict": ("stable" if value < 0.10
                        else "watch" if value < 0.25
                        else "investigate / retrain"),
        })
    return pd.DataFrame(rows)


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed bad rate by probability decile.

    Ranking and calibration are different properties. A policy that sets
    thresholds on probability needs both: good ranking with bad calibration
    means the chosen cut-off does not deliver the bad rate it promises.
    """
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int),
                       "p": np.asarray(y_prob, dtype=float)})
    df["decile"] = pd.qcut(df["p"], q=bins, duplicates="drop", labels=False)

    out = (
        df.groupby("decile")
        .agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
        .reset_index()
    )
    out["gap"] = out["observed"] - out["predicted"]
    for col in ("predicted", "observed", "gap"):
        out[col] = out[col].round(4)
    return out
