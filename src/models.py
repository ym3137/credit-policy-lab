"""Three tiers of increasing complexity. Module M2.

The comparison exists to answer one question: does the extra complexity change
the POLICY, or only the AUC? Those are different claims, and only the first is
worth paying for in interpretability and maintenance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


def simple_rule_score(df: pd.DataFrame) -> np.ndarray:
    """Tier 1 — the baseline any analyst could build in an afternoon.

    FICO midpoint and DTI only, standardised and summed. If a gradient-boosted
    model on 40 features cannot beat this on the policy curve, that is the
    finding, and it is a more useful one than a marginal AUC gain.
    """
    fico = (df["fico_range_low"].astype(float) + df["fico_range_high"].astype(float)) / 2
    dti = pd.to_numeric(df["dti"], errors="coerce")

    z_fico = (fico - fico.mean()) / fico.std()
    z_dti = (dti - dti.mean()) / dti.std()
    return (-z_fico + z_dti.fillna(0)).to_numpy()   # higher = riskier


def fit_gbm(X_train: pd.DataFrame, y_train: pd.Series,
            random_state: int = 42) -> HistGradientBoostingClassifier:
    """Tier 3 — handles NaN natively, so no imputation step to leak through."""
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4,
        early_stopping=True, validation_fraction=0.15,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model
