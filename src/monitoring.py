"""Post-deployment monitoring simulation. Module M4.

Answers the operational question a risk team actually lives with: how long
until this model needs retraining?
"""
from __future__ import annotations

import pandas as pd

from .metrics import psi, summary_metrics


def performance_by_period(df: pd.DataFrame, score_col: str,
                          target_col: str = "target",
                          date_col: str = "issue_dt", freq: str = "Q") -> pd.DataFrame:
    """KS / AUC per issue cohort — the decay curve.

    Only cohorts with resolved labels appear. The unlabelled drift window is
    handled by `score_drift` instead, which needs no target.
    """
    work = df[df[target_col].notna()].copy()
    rows = []
    for period, part in work.groupby(work[date_col].dt.to_period(freq)):
        if part[target_col].nunique() < 2:
            continue
        metrics = summary_metrics(part[target_col].to_numpy(),
                                  part[score_col].to_numpy())
        metrics["period"] = str(period)
        rows.append(metrics)
    cols = ["period", "n", "bad_rate", "auc", "ks", "capture_top_10pct"]
    return pd.DataFrame(rows)[cols]


def score_drift(baseline_scores: pd.Series, df: pd.DataFrame, score_col: str,
                date_col: str = "issue_dt", freq: str = "Q") -> pd.DataFrame:
    """PSI of the score distribution per period against the training baseline.

    This is what makes the 2016-2018 window useful despite its incomplete
    labels: measuring distribution shift requires no outcome at all.
    """
    rows = []
    for period, part in df.groupby(df[date_col].dt.to_period(freq)):
        value = psi(baseline_scores.to_numpy(), part[score_col].to_numpy())
        rows.append({
            "period": str(period),
            "n": len(part),
            "psi": round(value, 4),
            "verdict": ("stable" if value < 0.10
                        else "watch" if value < 0.25
                        else "investigate / retrain"),
        })
    return pd.DataFrame(rows)


def retraining_signal(decay: pd.DataFrame, drift: pd.DataFrame,
                      ks_drop_tolerance: float = 0.20) -> pd.DataFrame:
    """Combine decay and drift into a single retrain flag per period.

    `ks_drop_tolerance` is a RELATIVE drop from the first observed cohort — a
    20% fall from KS 0.35 to 0.28, not an absolute floor. Absolute thresholds
    travel badly between portfolios.
    """
    merged = decay.merge(drift[["period", "psi", "verdict"]], on="period", how="outer")
    merged = merged.sort_values("period").reset_index(drop=True)

    if "ks" in merged and merged["ks"].notna().any():
        baseline_ks = merged["ks"].dropna().iloc[0]
        merged["ks_relative_drop"] = ((baseline_ks - merged["ks"]) / baseline_ks).round(4)
        merged["retrain_flag"] = (
            (merged["ks_relative_drop"] > ks_drop_tolerance)
            | (merged["psi"] > 0.25)
        )
    return merged
