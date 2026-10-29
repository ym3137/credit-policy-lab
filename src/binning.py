"""WOE binning and IV — the scorecard foundation.

Convention used throughout (state it, because the sign flips between texts):

    WOE_i = ln( (bad_i / bad_total) / (good_i / good_total) )

so a POSITIVE WOE means the bin is riskier than the population average.

    IV = sum_i (bad_share_i - good_share_i) * WOE_i
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 0.5          # Haldane-style correction for empty cells
MISSING_LABEL = "__MISSING__"

IV_BANDS = [
    (0.02, "no signal — drop"),
    (0.10, "weak"),
    (0.30, "medium — scorecard workhorse"),
    (0.50, "strong"),
    (np.inf, "SUSPECT — check for leakage before celebrating"),
]


def iv_verdict(iv: float) -> str:
    for upper, label in IV_BANDS:
        if iv < upper:
            return label
    return IV_BANDS[-1][1]


def _quantile_edges(x: pd.Series, max_bins: int) -> np.ndarray:
    """Quantile cut points, de-duplicated. Spiky/zero-inflated columns collapse
    to fewer bins on their own, which is the correct behaviour."""
    qs = np.linspace(0, 1, max_bins + 1)
    edges = np.unique(np.nanquantile(x.astype(float), qs))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def bin_numeric(x: pd.Series, max_bins: int = 6) -> pd.Series:
    """Cut a numeric column into bins; NaN becomes its own labelled bin.

    Missing is never imputed. On credit data 'no record' is usually information
    (a thin file), not an inconvenience — give it a bin and let the WOE say so.
    """
    edges = _quantile_edges(x, max_bins)
    binned = pd.cut(x.astype(float), bins=edges, duplicates="drop")
    out = binned.astype(object)
    out[x.isna()] = MISSING_LABEL
    return pd.Series(out, index=x.index)


def woe_table(bins: pd.Series, target: pd.Series) -> pd.DataFrame:
    """Per-bin counts, bad rate, WOE and IV contribution."""
    df = pd.DataFrame({"bin": bins, "target": target.astype(int)})
    grp = df.groupby("bin", dropna=False, observed=True)["target"]

    tab = pd.DataFrame({"count": grp.size(), "bads": grp.sum()}).reset_index()
    tab["goods"] = tab["count"] - tab["bads"]

    bad_total, good_total = tab["bads"].sum(), tab["goods"].sum()
    if bad_total == 0 or good_total == 0:
        raise ValueError("target has only one class — WOE is undefined")

    tab["bad_rate"] = tab["bads"] / tab["count"]
    tab["count_share"] = tab["count"] / tab["count"].sum()
    tab["bad_share"] = (tab["bads"] + EPS) / (bad_total + EPS * len(tab))
    tab["good_share"] = (tab["goods"] + EPS) / (good_total + EPS * len(tab))

    tab["woe"] = np.log(tab["bad_share"] / tab["good_share"])
    tab["iv_contrib"] = (tab["bad_share"] - tab["good_share"]) * tab["woe"]

    # Sort real bins by their interval, keep MISSING at the end.
    is_missing = tab["bin"].astype(str) == MISSING_LABEL
    real = tab[~is_missing].copy()
    real["_left"] = [b.left if hasattr(b, "left") else np.nan for b in real["bin"]]
    real = real.sort_values("_left").drop(columns="_left")
    tab = pd.concat([real, tab[is_missing]], ignore_index=True)

    for col in ("bad_rate", "count_share", "woe", "iv_contrib"):
        tab[col] = tab[col].round(4)
    return tab


def information_value(tab: pd.DataFrame) -> float:
    return float(tab["iv_contrib"].sum())


def is_monotonic(tab: pd.DataFrame) -> bool:
    """Does bad rate move in one direction across real bins?

    Non-monotonic is not automatically wrong — some variables genuinely turn
    (very high income can tick back up). It does mean the bins need a business
    reason, and that reason belongs in the data dictionary.
    """
    real = tab[tab["bin"].astype(str) != MISSING_LABEL]["bad_rate"].to_numpy()
    if len(real) < 3:
        return True
    diffs = np.diff(real)
    return bool(np.all(diffs >= 0) or np.all(diffs <= 0))


def merge_small_bins(tab: pd.DataFrame, min_frac: float = 0.05) -> pd.DataFrame:
    """Flag bins below the minimum share. Small bins produce unstable WOE that
    will not reproduce out-of-time, which is exactly where it matters."""
    tab = tab.copy()
    tab["below_min_share"] = tab["count_share"] < min_frac
    return tab


def profile_features(df: pd.DataFrame, features: list[str], target_col: str,
                     cfg: dict) -> pd.DataFrame:
    """IV, monotonicity and small-bin flags for every candidate feature.

    This is the feature-selection table. Sort by IV, then read the `verdict`
    column before the number — a top-of-table IV is a leakage suspect first and
    a strong predictor second.
    """
    bin_cfg = cfg["binning"]
    rows = []

    for feat in features:
        col = df[feat]
        try:
            if pd.api.types.is_numeric_dtype(col):
                bins = bin_numeric(col, max_bins=bin_cfg["max_bins"])
            else:
                bins = col.astype(object).where(col.notna(), MISSING_LABEL)
            tab = merge_small_bins(woe_table(bins, df[target_col]),
                                   bin_cfg["min_bin_frac"])
            iv = information_value(tab)
            rows.append({
                "feature": feat,
                "dtype": str(col.dtype),
                "fill_rate": round(float(col.notna().mean()), 4),
                "n_bins": len(tab),
                "iv": round(iv, 4),
                "verdict": iv_verdict(iv),
                "monotonic": is_monotonic(tab),
                "has_small_bin": bool(tab["below_min_share"].any()),
                "leakage_alarm": iv > bin_cfg["iv_leakage_alarm"],
                "error": None,
            })
        except Exception as exc:                      # noqa: BLE001
            rows.append({"feature": feat, "dtype": str(col.dtype), "iv": None,
                         "verdict": None, "error": str(exc)})

    return pd.DataFrame(rows).sort_values("iv", ascending=False, na_position="last")


def apply_woe(x: pd.Series, tab: pd.DataFrame, max_bins: int = 6) -> pd.Series:
    """Transform a raw column into its WOE values using a FITTED table.

    Bins and WOE values must come from train only. Re-fitting on validation or
    test leaks the target into the transform, which is a quieter version of the
    same mistake the leakage exclusion list exists to prevent.
    """
    mapping = dict(zip(tab["bin"].astype(str), tab["woe"]))
    if pd.api.types.is_numeric_dtype(x):
        bins = bin_numeric(x, max_bins=max_bins)
    else:
        bins = x.astype(object).where(x.notna(), MISSING_LABEL)
    return bins.astype(str).map(mapping).astype(float)
