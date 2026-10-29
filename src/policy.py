"""Approval-policy simulation — the module the whole repo exists for.

A probability is not a decision. This turns scores into an approval rule and
reports what that rule costs and buys: at a fixed approval rate, what bad rate
results, how many good applicants are turned away, and how the policy compares
against the platform's own grade ordering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def policy_at_approval_rate(y_true: np.ndarray, y_score: np.ndarray,
                            approval_rate: float) -> dict[str, float]:
    """Approve the safest `approval_rate` share; report the consequences.

    `y_score` is risk — higher means worse. The cut-off is a quantile of the
    score, so the approval rate is held exactly and the bad rate is the free
    variable. That is the direction the business question runs.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    n_approved = max(1, int(round(n * approval_rate)))

    order = np.argsort(y_score)             # safest first
    approved = order[:n_approved]
    declined = order[n_approved:]

    total_bads = int(y_true.sum())
    total_goods = n - total_bads
    approved_bads = int(y_true[approved].sum())
    declined_bads = int(y_true[declined].sum()) if len(declined) else 0
    declined_goods = len(declined) - declined_bads

    return {
        "approval_rate": round(approval_rate, 4),
        "cutoff_score": round(float(y_score[order[n_approved - 1]]), 6),
        "n_approved": n_approved,
        "approved_bad_rate": round(approved_bads / n_approved, 4),
        "bads_avoided_share": round(declined_bads / total_bads, 4) if total_bads else np.nan,
        "goods_rejected_share": round(declined_goods / total_goods, 4) if total_goods else np.nan,
        # Among everyone declined, how many would actually have repaid.
        # This is the customer-experience cost of tightening, and it is always
        # the larger number — say so rather than reporting avoided bads alone.
        "precision_of_decline": round(declined_bads / len(declined), 4) if len(declined) else np.nan,
    }


def policy_curve(y_true: np.ndarray, y_score: np.ndarray,
                 approval_rates: list[float]) -> pd.DataFrame:
    """The trade-off curve: one row per approval rate. The deliverable chart."""
    return pd.DataFrame([
        policy_at_approval_rate(y_true, y_score, rate) for rate in approval_rates
    ])


def compare_policies(y_true: np.ndarray, scores: dict[str, np.ndarray],
                     approval_rates: list[float]) -> pd.DataFrame:
    """Several scoring approaches on the same axis.

    Pass the platform's own ordering (grade / sub_grade rank) in as one of the
    scores. At an identical approval rate the comparison is like-for-like and
    needs no normalisation — that is why `grade` is held out of the feature set.
    """
    frames = []
    for name, score in scores.items():
        part = policy_curve(y_true, score, approval_rates)
        part.insert(0, "policy", name)
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def grade_to_score(grade: pd.Series) -> pd.Series:
    """Turn LendingClub's A1..G5 grade into an ordinal risk score.

    Ties inside a grade are preserved — the platform genuinely does not
    distinguish within a sub-grade, so the benchmark should not pretend it
    does. Expect its policy curve to look like a step function.
    """
    clean = grade.astype(str).str.strip().str.upper()
    letters = sorted({g[0] for g in clean if g and g[0].isalpha()})
    letter_rank = {ltr: i for i, ltr in enumerate(letters)}

    def rank(value: str) -> float:
        if not value or not value[0].isalpha():
            return np.nan
        base = letter_rank.get(value[0], np.nan) * 10
        digits = "".join(ch for ch in value[1:] if ch.isdigit())
        return base + (int(digits) if digits else 0)

    return clean.map(rank)


def rough_economics(df: pd.DataFrame, approved_mask: np.ndarray, cfg: dict,
                    amount_col: str = "loan_amnt",
                    rate_col: str = "int_rate",
                    target_col: str = "target") -> dict[str, float]:
    """Order-of-magnitude return on an approved book.

    DELIBERATELY CRUDE, and the README says so. No funding cost, no servicing
    cost, no prepayment, no collections model, and a flat assumed recovery rate.
    It ranks policies against each other on the same assumptions; it does not
    estimate profit, and no number from it belongs in a resume bullet.
    """
    assumed_recovery = float(cfg["policy"]["assumed_recovery_rate"])
    book = df.loc[approved_mask].copy()

    rate = book[rate_col]
    if rate.dtype == object:
        rate = rate.astype(str).str.replace("%", "", regex=False).astype(float)
    rate = rate / 100.0

    amount = book[amount_col].astype(float)
    is_bad = book[target_col].astype(int) == 1

    interest = (amount * rate).where(~is_bad, 0.0).sum()
    principal_lost = (amount * (1 - assumed_recovery)).where(is_bad, 0.0).sum()
    deployed = amount.sum()

    return {
        "n_approved": int(len(book)),
        "capital_deployed": round(float(deployed), 2),
        "gross_interest_proxy": round(float(interest), 2),
        "principal_loss_proxy": round(float(principal_lost), 2),
        "net_proxy": round(float(interest - principal_lost), 2),
        "net_proxy_pct_deployed": round(float((interest - principal_lost) / deployed), 4)
        if deployed else np.nan,
        "assumed_recovery_rate": assumed_recovery,
    }
