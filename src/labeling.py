"""Target construction and the performance window.

The whole point of this module is that `loan_status` has THREE groups, not two.
Censored loans (still Current, or merely late) have not had the opportunity to
default. Mapping them to good imports future defaults into the good class and
inflates every metric downstream, so they are dropped and counted, loudly.
"""
from __future__ import annotations

import pandas as pd

ISSUE_DATE_FORMAT = "%b-%Y"   # e.g. "Dec-2015"


def parse_issue_date(series: pd.Series) -> pd.Series:
    """`issue_d` arrives as 'Dec-2015'. Return a month-start timestamp."""
    return pd.to_datetime(series.str.strip(), format=ISSUE_DATE_FORMAT, errors="coerce")


def parse_term_months(series: pd.Series) -> pd.Series:
    """`term` arrives as ' 36 months' — leading space is real. Return an int."""
    return (
        series.astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype("Float64")
        .astype("Int64")
    )


def assign_target(status: pd.Series, labels_cfg: dict) -> pd.Series:
    """Map loan_status to {1: bad, 0: good, <NA>: censored}.

    Any status not listed in the config lands as <NA> too, so a new or
    misspelled value fails visibly instead of silently becoming good.
    """
    bad = set(labels_cfg["bad"])
    good = set(labels_cfg["good"])

    target = pd.Series(pd.NA, index=status.index, dtype="Int64")
    clean = status.astype(str).str.strip()
    target[clean.isin(bad)] = 1
    target[clean.isin(good)] = 0
    return target


def build_modelling_sample(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply term filter, issue window and label resolution.

    Returns (sample, audit) where `audit` is the row-count waterfall. Always
    print the waterfall — the drop from 2.26M to the final sample is a number
    a reviewer will ask about, and it should be answerable in one table.
    """
    sample_cfg = cfg["sample"]
    steps: list[dict] = []

    work = df.copy()
    steps.append({"step": "raw rows", "rows": len(work)})

    work["issue_dt"] = parse_issue_date(work["issue_d"])
    work["term_months"] = parse_term_months(work["term"])

    work = work[work["issue_dt"].notna()]
    steps.append({"step": "parseable issue_d", "rows": len(work)})

    work = work[work["term_months"] == sample_cfg["term_months"]]
    steps.append({
        "step": f"term == {sample_cfg['term_months']} months",
        "rows": len(work),
    })

    start = pd.Timestamp(sample_cfg["issue_start"])
    end = pd.Timestamp(sample_cfg["issue_end"])
    work = work[(work["issue_dt"] >= start) & (work["issue_dt"] <= end)]
    steps.append({
        "step": f"issued {start:%Y-%m} .. {end:%Y-%m}",
        "rows": len(work),
    })

    work["target"] = assign_target(work["loan_status"], cfg["labels"])
    n_censored = int(work["target"].isna().sum())
    work = work[work["target"].notna()].copy()
    work["target"] = work["target"].astype(int)
    steps.append({
        "step": f"resolved status (dropped {n_censored:,} censored)",
        "rows": len(work),
    })

    audit = pd.DataFrame(steps)
    audit["pct_of_raw"] = (audit["rows"] / audit.loc[0, "rows"] * 100).round(2)
    return work, audit


def check_window_sufficiency(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per issue-quarter: did every loan have a full term plus buffer to resolve?

    A quarter whose censored share is materially above zero means the window
    was cut too close to the data cutoff — the filter is wrong, not the data.
    """
    sample_cfg = cfg["sample"]
    cutoff = pd.Timestamp(sample_cfg["data_cutoff"])
    term = int(sample_cfg["term_months"])
    buffer = int(sample_cfg["buffer_months"])

    work = df.copy()
    if "issue_dt" not in work:
        work["issue_dt"] = parse_issue_date(work["issue_d"])

    work["matures_dt"] = work["issue_dt"] + pd.DateOffset(months=term)
    work["months_observable"] = (
        (cutoff.year - work["issue_dt"].dt.year) * 12
        + (cutoff.month - work["issue_dt"].dt.month)
    )
    work["has_full_window"] = work["months_observable"] >= (term + buffer)
    work["is_censored"] = assign_target(work["loan_status"], cfg["labels"]).isna()

    out = (
        work.groupby(work["issue_dt"].dt.to_period("Q"))
        .agg(
            loans=("issue_dt", "size"),
            months_observable=("months_observable", "min"),
            full_window_share=("has_full_window", "mean"),
            censored_share=("is_censored", "mean"),
        )
        .reset_index()
        .rename(columns={"issue_dt": "issue_quarter"})
    )
    out["full_window_share"] = out["full_window_share"].round(4)
    out["censored_share"] = out["censored_share"].round(4)
    return out
