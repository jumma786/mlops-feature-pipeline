"""
Feature Engineering Pipeline — Versioned Sklearn Transformers
==============================================================
Each transformer is:
  - Leakage-safe (fit on train, transform test)
  - MLflow-logged as a versioned artifact
  - Unit-testable in isolation
  - Decoupled from the model — feature pipeline can be retrained independently

Transformers:
  1. DurationDropper          — removes leakage feature
  2. CategoricalEncoder       — label encodes all object columns
  3. EconomicFeatureEngineer  — creates euribor_emp_interaction
  4. CampaignIntensityEncoder — bins campaign count into intensity levels
  5. ContactRecencyEncoder    — encodes pdays into recency buckets
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class DurationDropper(BaseEstimator, TransformerMixin):
    """
    Drops 'duration' feature — known leakage risk.
    Call duration is unknown before the call is made.
    Reference: Moro et al., 2014 (UCI Bank Marketing paper).
    """
    def __init__(self, col="duration"):
        self.col = col

    def fit(self, X, y=None):
        self.dropped_ = self.col in X.columns
        return self

    def transform(self, X):
        X = X.copy()
        if self.dropped_ and self.col in X.columns:
            X = X.drop(columns=[self.col])
            logger.info(f"Dropped leakage feature: {self.col}")
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return [f for f in input_features if f != self.col]


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Label-encodes all object/categorical columns.
    Fit learns the encoding on train set — applied consistently to test.
    """
    def __init__(self):
        self.encoders_ = {}
        self.cat_cols_ = []

    def fit(self, X, y=None):
        self.cat_cols_ = X.select_dtypes(include="object").columns.tolist()
        for col in self.cat_cols_:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders_[col] = le
        logger.info(f"Fitted encoders for {len(self.cat_cols_)} categorical columns")
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cat_cols_:
            if col in X.columns:
                le = self.encoders_[col]
                # Handle unseen categories
                X[col] = X[col].astype(str).map(
                    lambda x, le=le: le.transform([x])[0]
                    if x in le.classes_ else -1
                )
        return X


class EconomicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates interaction feature: euribor_emp_interaction.
    Captures combined economic signal — low euribor + low employment
    variance = economic downturn = higher subscription likelihood.
    """
    def __init__(self):
        self.feature_name_ = "euribor_emp_interaction"

    def fit(self, X, y=None):
        required = {"euribor3m", "emp.var.rate"}
        self.available_ = required.issubset(set(X.columns))
        if not self.available_:
            logger.warning("euribor3m or emp.var.rate not found — skipping interaction")
        return self

    def transform(self, X):
        X = X.copy()
        if self.available_:
            X[self.feature_name_] = X["euribor3m"] * X["emp.var.rate"]
            logger.info(f"Created feature: {self.feature_name_}")
        return X


class CampaignIntensityEncoder(BaseEstimator, TransformerMixin):
    """
    Bins 'campaign' (number of contacts) into intensity levels:
    1 contact = low, 2-3 = medium, 4-6 = high, 7+ = very_high.
    Too many contacts = customer fatigue = lower subscription rate.
    """
    def __init__(self, col="campaign"):
        self.col = col
        self.new_col = "campaign_intensity"

    def fit(self, X, y=None):
        self.available_ = self.col in X.columns
        return self

    def transform(self, X):
        X = X.copy()
        if self.available_ and self.col in X.columns:
            conditions = [
                X[self.col] == 1,
                X[self.col].between(2, 3),
                X[self.col].between(4, 6),
                X[self.col] >= 7,
            ]
            choices = [0, 1, 2, 3]  # low, medium, high, very_high
            X[self.new_col] = np.select(conditions, choices, default=1)
            logger.info(f"Created feature: {self.new_col}")
        return X


class ContactRecencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes 'pdays' (days since last contact) into recency buckets:
    999 = never contacted, 1-7 = very recent, 8-30 = recent, 30+ = old.
    pdays=999 means client was never previously contacted.
    """
    def __init__(self, col="pdays"):
        self.col = col
        self.new_col = "contact_recency"

    def fit(self, X, y=None):
        self.available_ = self.col in X.columns
        return self

    def transform(self, X):
        X = X.copy()
        if self.available_ and self.col in X.columns:
            conditions = [
                X[self.col] == 999,
                X[self.col].between(1, 7),
                X[self.col].between(8, 30),
                X[self.col] > 30,
            ]
            choices = [0, 3, 2, 1]  # never, very_recent, recent, old
            X[self.new_col] = np.select(conditions, choices, default=0)
            logger.info(f"Created feature: {self.new_col}")
        return X
