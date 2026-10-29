"""Logistic-regression scorecard on WOE-transformed features. Module M1."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .binning import (MISSING_LABEL, apply_woe, bin_numeric, compute_edges,
                      information_value, woe_table)


class Scorecard:
    """Fit bins, WOE and coefficients on TRAIN ONLY, then apply frozen.

    Bin EDGES are stored alongside the WOE tables. Re-deriving edges on
    validation would give each split its own cut-points, so the fitted WOE
    values would be attached to different populations than they were measured
    on — the transform itself would leak.
    """

    def __init__(self, features: list[str], max_bins: int = 6,
                 base_score: int = 600, pdo: int = 20, base_odds: float = 50.0):
        self.features = list(features)
        self.max_bins = max_bins
        self.base_score, self.pdo, self.base_odds = base_score, pdo, base_odds
        self.edges: dict[str, np.ndarray | None] = {}
        self.woe_tables: dict[str, pd.DataFrame] = {}
        self.iv: dict[str, float] = {}
        self.model: LogisticRegression | None = None

    # ---------------------------------------------------------------- fit
    def fit(self, df: pd.DataFrame, target_col: str = "target") -> "Scorecard":
        y = df[target_col].astype(int)
        woe_frame = {}

        for feat in self.features:
            col = df[feat]
            if pd.api.types.is_numeric_dtype(col):
                self.edges[feat] = compute_edges(col, self.max_bins)
                bins = bin_numeric(col, self.max_bins, edges=self.edges[feat])
            else:
                self.edges[feat] = None
                bins = col.astype(object).where(col.notna(), MISSING_LABEL)

            tab = woe_table(bins, y)
            self.woe_tables[feat] = tab
            self.iv[feat] = information_value(tab)
            woe_frame[feat] = apply_woe(col, tab, self.max_bins, self.edges[feat])

        X = pd.DataFrame(woe_frame).fillna(0.0)
        self.model = LogisticRegression(max_iter=2000, solver="lbfgs")
        self.model.fit(X, y)
        return self

    # ---------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.woe_tables:
            raise RuntimeError("fit() first")
        return pd.DataFrame({
            f: apply_woe(df[f], self.woe_tables[f], self.max_bins, self.edges[f])
            for f in self.features
        }, index=df.index).fillna(0.0)

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

    # ------------------------------------------------------------- output
    def coefficient_table(self) -> pd.DataFrame:
        """Per-feature coefficient and IV.

        A NEGATIVE coefficient on a WOE feature is a red flag: WOE is already
        oriented so that higher means riskier, so the fitted sign should be
        positive. A negative one usually means that feature is correlated with
        another and the model is using it to cancel, which will not reproduce
        out-of-time.
        """
        if self.model is None:
            raise RuntimeError("fit() first")
        out = pd.DataFrame({
            "feature": self.features,
            "coefficient": np.round(self.model.coef_[0], 4),
            "iv": [round(self.iv[f], 4) for f in self.features],
        })
        out["sign_ok"] = out["coefficient"] > 0
        return out.sort_values("iv", ascending=False)

    def points_table(self) -> pd.DataFrame:
        """The auditable artefact: per-bin points.

        This table — not the AUC — is what an underwriter executes, a compliance
        reviewer audits, and an analyst traces when a declined applicant asks
        why.
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
