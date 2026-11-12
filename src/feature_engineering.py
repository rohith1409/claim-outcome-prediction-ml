from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PayerFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds payer denial rate and claim count features using training-only statistics."""

    def __init__(self):
        self.payer_denial_rates_ = {}
        self.payer_claim_counts_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        if y is None:
            raise ValueError("PayerFeatureEngineer.fit requires the target to compute denial rates.")

        train_df = X.copy()
        train_df["status"] = y.reset_index(drop=True)
        self.payer_denial_rates_ = (
            train_df.groupby("payer")["status"].apply(lambda s: (s == "Denied").mean()).to_dict()
        )
        self.payer_claim_counts_ = train_df.groupby("payer").size().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["payer_denial_rate"] = X["payer"].map(self.payer_denial_rates_).fillna(0.0)
        X["payer_claim_count"] = X["payer"].map(self.payer_claim_counts_).fillna(0)
        return X
