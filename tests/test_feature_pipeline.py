"""
Test suite for mlops-feature-pipeline.
Run: pytest tests/ -v --cov=src
"""

import pytest
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.transformers import (
    DurationDropper, CategoricalEncoder, EconomicFeatureEngineer,
    CampaignIntensityEncoder, ContactRecencyEncoder,
)
from src.features.pipeline_builder import build_feature_pipeline
from src.data.generator import generate_synthetic


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    X_train, X_test, y_train, y_test = generate_synthetic(n_samples=300)
    return X_train, X_test, y_train, y_test


# ── DurationDropper Tests ─────────────────────────────────────────────────────

class TestDurationDropper:
    def test_drops_duration(self, sample_data):
        X_train, X_test, _, _ = sample_data
        t = DurationDropper()
        t.fit(X_train)
        X_out = t.transform(X_train)
        assert "duration" not in X_out.columns

    def test_other_columns_preserved(self, sample_data):
        X_train, _, _, _ = sample_data
        t = DurationDropper()
        t.fit(X_train)
        X_out = t.transform(X_train)
        assert "age" in X_out.columns
        assert "campaign" in X_out.columns

    def test_fit_transform_consistent(self, sample_data):
        X_train, X_test, _, _ = sample_data
        t = DurationDropper()
        t.fit(X_train)
        out_train = t.transform(X_train)
        out_test  = t.transform(X_test)
        assert out_train.shape[1] == out_test.shape[1]


# ── CategoricalEncoder Tests ──────────────────────────────────────────────────

class TestCategoricalEncoder:
    def test_no_object_columns_after_transform(self, sample_data):
        X_train, X_test, _, _ = sample_data
        t = CategoricalEncoder()
        t.fit(X_train)
        X_out = t.transform(X_train)
        assert X_out.select_dtypes(include="object").shape[1] == 0

    def test_fit_on_train_transform_test(self, sample_data):
        X_train, X_test, _, _ = sample_data
        t = CategoricalEncoder()
        t.fit(X_train)
        X_out = t.transform(X_test)
        assert X_out.select_dtypes(include="object").shape[1] == 0

    def test_shape_preserved(self, sample_data):
        X_train, _, _, _ = sample_data
        t = CategoricalEncoder()
        t.fit(X_train)
        X_out = t.transform(X_train)
        assert X_out.shape[0] == X_train.shape[0]


# ── EconomicFeatureEngineer Tests ─────────────────────────────────────────────

class TestEconomicFeatureEngineer:
    def test_creates_interaction_feature(self, sample_data):
        X_train, _, _, _ = sample_data
        t = EconomicFeatureEngineer()
        t.fit(X_train)
        X_out = t.transform(X_train)
        assert "euribor_emp_interaction" in X_out.columns

    def test_interaction_values_correct(self, sample_data):
        X_train, _, _, _ = sample_data
        t = EconomicFeatureEngineer()
        t.fit(X_train)
        X_out = t.transform(X_train)
        expected = X_train["euribor3m"] * X_train["emp.var.rate"]
        pd.testing.assert_series_equal(
            X_out["euribor_emp_interaction"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )


# ── CampaignIntensityEncoder Tests ────────────────────────────────────────────

class TestCampaignIntensityEncoder:
    def test_creates_intensity_feature(self, sample_data):
        X_train, _, _, _ = sample_data
        t = CampaignIntensityEncoder()
        t.fit(X_train)
        X_out = t.transform(X_train)
        assert "campaign_intensity" in X_out.columns

    def test_intensity_values_in_range(self, sample_data):
        X_train, _, _, _ = sample_data
        t = CampaignIntensityEncoder()
        t.fit(X_train)
        X_out = t.transform(X_train)
        assert X_out["campaign_intensity"].between(0, 3).all()


# ── ContactRecencyEncoder Tests ───────────────────────────────────────────────

class TestContactRecencyEncoder:
    def test_creates_recency_feature(self, sample_data):
        X_train, _, _, _ = sample_data
        t = ContactRecencyEncoder()
        t.fit(X_train)
        X_out = t.transform(X_train)
        assert "contact_recency" in X_out.columns

    def test_never_contacted_encoded_as_zero(self, sample_data):
        X_train, _, _, _ = sample_data
        t = ContactRecencyEncoder()
        t.fit(X_train)
        X_out = t.transform(X_train)
        never_mask = X_train["pdays"] == 999
        assert (X_out.loc[never_mask, "contact_recency"] == 0).all()


# ── Full Pipeline Tests ───────────────────────────────────────────────────────

class TestFullPipeline:
    def test_pipeline_fit_transform(self, sample_data):
        X_train, X_test, _, _ = sample_data
        pipeline = build_feature_pipeline()
        X_out = pipeline.fit_transform(X_train)
        assert X_out is not None

    def test_pipeline_adds_features(self, sample_data):
        X_train, _, _, _ = sample_data
        pipeline = build_feature_pipeline()
        X_out = pipeline.fit_transform(X_train)
        if isinstance(X_out, pd.DataFrame):
            assert X_out.shape[1] >= X_train.shape[1]

    def test_pipeline_no_object_columns(self, sample_data):
        X_train, _, _, _ = sample_data
        pipeline = build_feature_pipeline()
        X_out = pipeline.fit_transform(X_train)
        if isinstance(X_out, pd.DataFrame):
            assert X_out.select_dtypes(include="object").shape[1] == 0

    def test_pipeline_consistent_train_test(self, sample_data):
        X_train, X_test, _, _ = sample_data
        pipeline = build_feature_pipeline()
        X_train_out = pipeline.fit_transform(X_train)
        X_test_out  = pipeline.transform(X_test)
        if isinstance(X_train_out, np.ndarray):
            assert X_train_out.shape[1] == X_test_out.shape[1]
        else:
            assert X_train_out.shape[1] == X_test_out.shape[1]

    def test_pipeline_has_five_steps(self, sample_data):
        X_train, _, _, _ = sample_data
        pipeline = build_feature_pipeline()
        assert len(pipeline.steps) == 5
