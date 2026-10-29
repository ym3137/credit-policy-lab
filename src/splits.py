"""Out-of-time splits. There is deliberately no random-split function here.

A random hold-out lets the model train on 2015 applicants and be scored on
their contemporaries, which measures interpolation. A credit model is asked to
hold up on *next* quarter's applicants, so every split is on issue date.
"""
from __future__ import annotations

import pandas as pd

SPLIT_ORDER = ["train", "valid", "test", "drift"]


def assign_split(df: pd.DataFrame, cfg: dict, date_col: str = "issue_dt") -> pd.Series:
    """Label each row train / valid / test / drift by its issue date.

    Rows outside every configured window get <NA> rather than being quietly
    folded into the nearest split.
    """
    splits_cfg = cfg["splits"]
    out = pd.Series(pd.NA, index=df.index, dtype="object")

    for name in SPLIT_ORDER:
        if name not in splits_cfg:
            continue
        start, end = (pd.Timestamp(x) for x in splits_cfg[name])
        mask = (df[date_col] >= start) & (df[date_col] <= end)
        overlap = mask & out.notna()
        if overlap.any():
            raise ValueError(
                f"split '{name}' overlaps an earlier split on {int(overlap.sum())} rows "
                "— fix the windows in config.yaml"
            )
        out[mask] = name

    return out


def split_frames(df: pd.DataFrame, cfg: dict, date_col: str = "issue_dt") -> dict[str, pd.DataFrame]:
    """Return {'train': df, 'valid': df, 'test': df, 'drift': df}."""
    labels = assign_split(df, cfg, date_col=date_col)
    return {name: df[labels == name].copy() for name in SPLIT_ORDER if (labels == name).any()}


def split_summary(df: pd.DataFrame, cfg: dict, date_col: str = "issue_dt") -> pd.DataFrame:
    """Rows, date range and bad rate per split.

    Read the bad-rate column before modelling anything. If train and test bad
    rates differ sharply, the out-of-time gap is partly a shift in the
    population, and model-performance decay later cannot be attributed to the
    model alone.
    """
    work = df.copy()
    work["split"] = assign_split(work, cfg, date_col=date_col)

    rows = []
    for name in SPLIT_ORDER:
        part = work[work["split"] == name]
        if part.empty:
            continue
        has_target = "target" in part and part["target"].notna().any()
        rows.append({
            "split": name,
            "rows": len(part),
            "first_issue": part[date_col].min().strftime("%Y-%m"),
            "last_issue": part[date_col].max().strftime("%Y-%m"),
            "bad_rate": round(float(part["target"].mean()), 4) if has_target else None,
        })

    unassigned = int(work["split"].isna().sum())
    if unassigned:
        rows.append({"split": "UNASSIGNED", "rows": unassigned,
                     "first_issue": None, "last_issue": None, "bad_rate": None})
    return pd.DataFrame(rows)


def vintage_table(df: pd.DataFrame, date_col: str = "issue_dt",
                  freq: str = "Q") -> pd.DataFrame:
    """Bad rate by issue cohort — the vintage curve.

    This is the first chart to produce and the last one to trust. A bad rate
    that moves across cohorts means the target itself is non-stationary, which
    caps how much out-of-time decay can fairly be blamed on the model.
    """
    work = df[df["target"].notna()].copy()
    out = (
        work.groupby(work[date_col].dt.to_period(freq))
        .agg(loans=("target", "size"), bads=("target", "sum"))
        .reset_index()
        .rename(columns={date_col: "issue_cohort"})
    )
    out["bad_rate"] = (out["bads"] / out["loans"]).round(4)
    return out
