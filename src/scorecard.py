"""Logistic-regression scorecard on WOE-transformed features. Module M1."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .binning import apply_woe, bin_numeric, woe_table, MISSING_LABEL


class Scorecard:
    """Fit bins and WOE on TRAIN ONLY, then apply the frozen mapping elsewhere.

    Re-fitting bins on validation or test leaks the target into the transform —
    a quieter version of the leakage the exclusion list exists to prevent.
    """

    def __init__(self, features: list[str], max_bins: int = 6,
                 base_score: int = 600, pdo: int = 20, base_odds: float = 50.0):
        self.features = features
        self.max_bins = max_bins
        self.base_score, self.pdo, self.base_odds = base_score, pdo, base_odds
        self.woe_tables: dict[str, pd.DataFrame] = {}
        self.model: LogisticRegression | None = None

    def fit(self, df: pd.DataFrame, target_col: str = "target") -> "Scorecard":
        y = df[target_col].astype(int)
        woe_frame = {}
        for feat in self.features:
            col = df[feat]
            bins = (bin_numeric(col, self.max_bins)
                    if pd.api.types.is_numeric_dtype(col)
                    else col.astype(object).where(col.notna(), MISSING_LABEL))
            self.woe_tables[feat] = woe_table(bins, y)
            woe_frame[feat] = apply_woe(col, self.woe_tables[feat], self.max_bins)

        X = pd.DataFrame(woe_frame).fillna(0.0)
        self.model = LogisticRegression(max_iter=1000, solver="lbfgs")
        self.model.fit(X, y)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            f: apply_woe(df[f], self.woe_tables[f], self.max_bins)
            for f in self.features
        }).fillna(0.0)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit() first")
        return self.model.predict_proba(self.transform(df))[:, 1]

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Points, scaled so each `pdo` points halves the odds. Higher = safer."""
        p = np.clip(self.predict_proba(df), 1e-6, 1 - 1e-6)
        factor = self.pdo / np.log(2)
        offset = self.base_score - factor * np.log(self.base_odds)
        return offset + factor * np.log((1 - p) / p)

    def points_table(self) -> pd.DataFrame:
        """The auditable artefact: per-bin points.

        This table — not the AUC — is what an underwriter executes, a compliance
        reviewer audits, and an analyst traces when a decision is questioned.
        """
        if self.model is None:
            raise RuntimeError("fit() first")
        factor = self.pdo / np.log(2)
        coefs = dict(zip(self.features, self.model.coef_[0]))

        rows = []
        for feat, tab in self.woe_tables.items():
            for _, r in tab.iterrows():
                rows.append({
                    "feature": feat,
                    "bin": str(r["bin"]),
                    "count_share": r["count_share"],
                    "bad_rate": r["bad_rate"],
                    "woe": r["woe"],
                    "coefficient": round(float(coefs[feat]), 4),
                    "points": round(float(-coefs[feat] * r["woe"] * factor), 2),
                })
        return pd.DataFrame(rows)
