"""Field eligibility enforcement. See docs/leakage_fields.md for the reasoning.

Three categories, not two. The middle one — the platform's own decision — is
the subtle case: known at origination, so not leakage, but using it means
learning to imitate LendingClub's scorecard instead of assessing risk.
"""
from __future__ import annotations

import pandas as pd

# ---- Category A: generated after money goes out. Never a feature. ----------
POST_ORIGINATION = [
    "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "last_credit_pull_d",
    "last_fico_range_low", "last_fico_range_high",
    "debt_settlement_flag", "debt_settlement_flag_date",
    "settlement_status", "settlement_date", "settlement_amount",
    "settlement_percentage", "settlement_term",
    "pymnt_plan", "funded_amnt_inv",
]
POST_ORIGINATION_PREFIXES = ["hardship_", "settlement_", "deferral_"]

# ---- Category B: LendingClub's own decision. Benchmark only. ---------------
PLATFORM_DECISION = ["grade", "sub_grade", "int_rate", "installment"]

# ---- Always out: identifiers, free text, outcome. --------------------------
NON_FEATURES = [
    "id", "member_id", "url", "desc", "emp_title", "title", "zip_code",
    "loan_status", "issue_d", "issue_dt", "target", "term", "term_months",
    "policy_code",
]


def is_post_origination(col: str) -> bool:
    return col in POST_ORIGINATION or any(
        col.startswith(p) for p in POST_ORIGINATION_PREFIXES
    )


def eligible_features(columns: list[str]) -> dict[str, list[str]]:
    """Partition every column into eligible / platform / leaked / excluded."""
    buckets: dict[str, list[str]] = {
        "eligible": [], "platform_decision": [], "post_origination": [], "excluded": [],
    }
    for col in columns:
        if is_post_origination(col):
            buckets["post_origination"].append(col)
        elif col in PLATFORM_DECISION:
            buckets["platform_decision"].append(col)
        elif col in NON_FEATURES:
            buckets["excluded"].append(col)
        else:
            buckets["eligible"].append(col)
    return buckets


def leakage_report(profile: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Flag suspiciously predictive features from `binning.profile_features`.

    A feature with IV above the alarm threshold against a 36-month default
    target is almost never a discovery. It is almost always a category-A field
    that slipped past the exclusion list. Treat a hit as a bug report.
    """
    threshold = cfg["binning"]["iv_leakage_alarm"]
    flagged = profile[profile["iv"].fillna(0) > threshold].copy()
    flagged["suspected_post_origination"] = flagged["feature"].map(is_post_origination)
    flagged["action"] = flagged["suspected_post_origination"].map({
        True: "KNOWN LEAK — should already be excluded; check the pipeline",
        False: "INVESTIGATE — trace this field's timing before using it",
    })
    return flagged[["feature", "iv", "verdict", "suspected_post_origination", "action"]]


def availability_drift(df: pd.DataFrame, features: list[str],
                       date_col: str = "issue_dt", freq: str = "Y") -> pd.DataFrame:
    """Fill rate by issue period.

    A feature whose *availability* tracks the issue date will look predictive
    out-of-time for a reason that has nothing to do with credit risk. Several
    bureau aggregates in this file are near-empty before ~2015 and populated
    after. Drop or date-restrict them; do not let them into the scorecard.
    """
    period = df[date_col].dt.to_period(freq)
    fill = df[features].notna().groupby(period).mean().round(3)
    fill.index.name = "issue_period"

    spread = (fill.max() - fill.min()).rename("fill_rate_spread")
    out = fill.T.join(spread).sort_values("fill_rate_spread", ascending=False)
    out.index.name = "feature"
    return out.reset_index()
