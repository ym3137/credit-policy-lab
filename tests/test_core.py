"""Smoke tests on synthetic data — no raw file needed.

These verify the pipeline runs and that the metrics behave sanely at known
extremes (perfect separation, pure noise). They are not a substitute for
reading the real data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.binning import information_value, profile_features, woe_table, bin_numeric
from src.config import load_config
from src.labeling import assign_target, build_modelling_sample, check_window_sufficiency
from src.metrics import calibration_table, ks_statistic, psi, summary_metrics
from src.policy import compare_policies, grade_to_score, policy_curve
from src.splits import split_summary, vintage_table


def make_synthetic(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2012-01-01", "2018-12-01", freq="MS")
    issue = rng.choice(months, size=n)
    fico = rng.normal(695, 32, n).clip(620, 850)

    logit = -2.4 - 0.025 * (fico - 695) + rng.normal(0, 0.8, n)
    bad = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    issue_s = pd.Series(issue)
    resolved = issue_s <= pd.Timestamp("2015-12-01")
    status = np.where(resolved, np.where(bad == 1, "Charged Off", "Fully Paid"), "Current")

    return pd.DataFrame({
        "issue_d": issue_s.dt.strftime("%b-%Y"),
        "term": " 36 months",
        "loan_status": status,
        "fico_range_low": fico.round(0),
        "fico_range_high": (fico + 4).round(0),
        "dti": rng.normal(18, 8, n).clip(0, 45),
        "loan_amnt": rng.choice([5000, 10000, 15000, 20000], n),
        "int_rate": rng.uniform(6, 26, n).round(2),
        "grade": rng.choice(["A", "B", "C", "D", "E"], n, p=[.2, .3, .25, .15, .1]),
        "revol_util": rng.uniform(0, 100, n).round(1),
    })


def test_label_three_way_split():
    df = make_synthetic()
    cfg = load_config()
    target = assign_target(df["loan_status"], cfg["labels"])
    assert target.isna().sum() > 0, "Current loans must be censored, not good"
    assert set(target.dropna().unique()) <= {0, 1}
    print(f"  labels: {int((target==1).sum())} bad, {int((target==0).sum())} good, "
          f"{int(target.isna().sum())} censored")


def test_sample_waterfall_and_window():
    df = make_synthetic()
    cfg = load_config()
    sample, audit = build_modelling_sample(df, cfg)
    assert len(sample) > 0 and sample["target"].notna().all()
    assert sample["issue_dt"].max() <= pd.Timestamp(cfg["sample"]["issue_end"])
    window = check_window_sufficiency(df, cfg)
    assert (window["censored_share"] <= 1).all()
    print(audit.to_string(index=False))


def test_splits_are_disjoint_and_ordered():
    df = make_synthetic()
    cfg = load_config()
    sample, _ = build_modelling_sample(df, cfg)
    summary = split_summary(sample, cfg)
    assert "UNASSIGNED" not in summary["split"].values or \
        summary.loc[summary["split"] == "UNASSIGNED", "rows"].iloc[0] == 0
    train_end = summary.loc[summary["split"] == "train", "last_issue"].iloc[0]
    test_start = summary.loc[summary["split"] == "test", "first_issue"].iloc[0]
    assert train_end < test_start, "test must start after train ends"
    print(summary.to_string(index=False))
    print(vintage_table(sample).tail(3).to_string(index=False))


def test_woe_sign_and_iv():
    df = make_synthetic()
    cfg = load_config()
    sample, _ = build_modelling_sample(df, cfg)
    bins = bin_numeric(sample["fico_range_low"], max_bins=5)
    tab = woe_table(bins, sample["target"])
    # FICO rises => risk falls => WOE must trend downward
    assert tab["woe"].iloc[0] > tab["woe"].iloc[-1], "WOE sign convention broken"
    iv = information_value(tab)
    assert iv > 0.05
    print(tab[["bin", "count", "bad_rate", "woe", "iv_contrib"]].to_string(index=False))
    print(f"  IV = {iv:.4f}")

    prof = profile_features(sample, ["fico_range_low", "dti", "revol_util"],
                            "target", cfg)
    print(prof[["feature", "iv", "verdict", "monotonic"]].to_string(index=False))


def test_metrics_at_extremes():
    y = np.array([0] * 500 + [1] * 100)
    assert abs(ks_statistic(y, y.astype(float)) - 1.0) < 1e-9, "perfect separation => KS 1"
    rng = np.random.default_rng(1)
    noise_ks = ks_statistic(y, rng.random(len(y)))
    assert noise_ks < 0.2, f"noise should give low KS, got {noise_ks}"
    assert psi(rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)) < 0.1
    assert psi(rng.normal(0, 1, 5000), rng.normal(1.5, 1, 5000)) > 0.25
    print(f"  KS perfect=1.0  noise={noise_ks:.4f}  PSI behaves correctly")


def test_policy_curve_and_benchmark():
    df = make_synthetic()
    cfg = load_config()
    sample, _ = build_modelling_sample(df, cfg)
    fico_mid = (sample["fico_range_low"] + sample["fico_range_high"]) / 2
    model_score = (-fico_mid).to_numpy()
    grade_score = grade_to_score(sample["grade"]).to_numpy()

    curve = policy_curve(sample["target"].to_numpy(), model_score,
                         cfg["policy"]["approval_rates"])
    rates = curve["approved_bad_rate"].to_numpy()
    assert np.all(np.diff(rates) >= -1e-9), "bad rate must rise as approval loosens"
    print(curve.to_string(index=False))

    cmp = compare_policies(sample["target"].to_numpy(),
                           {"fico_model": model_score, "platform_grade": grade_score},
                           [0.6, 0.8])
    print(cmp[["policy", "approval_rate", "approved_bad_rate",
               "bads_avoided_share", "goods_rejected_share"]].to_string(index=False))

    cal = calibration_table(sample["target"].to_numpy(),
                            1 / (1 + np.exp(-(model_score - model_score.mean()) / 30)))
    assert len(cal) > 5
    print(f"  calibration table: {len(cal)} deciles")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"\n=== {fn.__name__} ===")
        fn()
    print(f"\nAll {len(tests)} smoke tests passed.")
