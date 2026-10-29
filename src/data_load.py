"""Loading and profiling the raw files. Module M0."""
from __future__ import annotations

import pandas as pd

from .labeling import parse_issue_date


def load_accepted(path: str, usecols: list[str] | None = None,
                  nrows: int | None = None) -> pd.DataFrame:
    """Read the accepted file. ~2.26M rows x 151 columns — pass `usecols`.

    `low_memory=False` avoids mixed-dtype warnings on the columns that hold
    both numbers and sentinel strings.
    """
    return pd.read_csv(path, usecols=usecols, nrows=nrows, low_memory=False)


def load_rejected(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Read the rejected file (~27.6M rows, 9 columns) and normalise names."""
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns
    ]
    for col in ("debt_to_income_ratio",):
        if col in df:
            df[col] = (df[col].astype(str).str.replace("%", "", regex=False)
                       .replace({"nan": None}).astype(float))
    return df


def clean_percent(series: pd.Series) -> pd.Series:
    """`revol_util` and `int_rate` arrive as strings with a % sign."""
    if series.dtype == object:
        return (series.astype(str).str.replace("%", "", regex=False)
                .replace({"nan": None, "": None}).astype(float))
    return series.astype(float)


def months_of_credit_history(earliest_cr_line: pd.Series,
                             issue_dt: pd.Series) -> pd.Series:
    """Credit history length at application, in months.

    The raw `earliest_cr_line` date must never be used directly: its absolute
    value is correlated with the issue date, so it smuggles vintage into the
    feature set. The elapsed duration is the actual risk signal.
    """
    first = pd.to_datetime(earliest_cr_line.astype(str).str.strip(),
                           format="%b-%Y", errors="coerce")
    return ((issue_dt.dt.year - first.dt.year) * 12
            + (issue_dt.dt.month - first.dt.month)).astype("Float64")


def profile_columns(df: pd.DataFrame, by_quarter: bool = True) -> pd.DataFrame:
    """Per-column dtype, fill rate, cardinality — and fill rate by issue quarter.

    The quarterly view is the part that matters: it is how availability drift
    gets caught before a feature reaches the scorecard.
    """
    rows = []
    for col in df.columns:
        series = df[col]
        rows.append({
            "column": col,
            "dtype": str(series.dtype),
            "fill_rate": round(float(series.notna().mean()), 4),
            "n_unique": int(series.nunique(dropna=True)),
            "sample_value": (series.dropna().iloc[0]
                             if series.notna().any() else None),
        })
    profile = pd.DataFrame(rows)

    if by_quarter and "issue_d" in df.columns:
        issue_dt = (df["issue_dt"] if "issue_dt" in df
                    else parse_issue_date(df["issue_d"]))
        quarters = issue_dt.dt.to_period("Q")
        fill_q = df.notna().groupby(quarters).mean()
        spread = (fill_q.max() - fill_q.min()).round(3)
        profile = profile.merge(
            spread.rename("fill_rate_spread_by_quarter").reset_index()
                  .rename(columns={"index": "column"}),
            on="column", how="left",
        )
    return profile.sort_values("fill_rate")
