"""
Feature Pipeline Builder
========================
Assembles all transformers into a versioned sklearn Pipeline.
Logs the fitted pipeline as an MLflow artifact — separate from the model.

Key principle: feature pipeline and model are DECOUPLED.
Either can be retrained independently without touching the other.
"""

import os
import sys
import logging
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features.transformers import (
    DurationDropper,
    CategoricalEncoder,
    EconomicFeatureEngineer,
    CampaignIntensityEncoder,
    ContactRecencyEncoder,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "mlops-feature-pipeline"
PIPELINE_NAME   = "FeaturePipelineV1"
ARTIFACT_PATH   = "feature_pipeline"


def build_feature_pipeline() -> Pipeline:
    """
    Builds the full feature engineering pipeline.
    Order matters — DurationDropper must come before CategoricalEncoder.
    """
    return Pipeline([
        ("drop_leakage",     DurationDropper()),
        ("encode_cats",      CategoricalEncoder()),
        ("economic_features", EconomicFeatureEngineer()),
        ("campaign_intensity", CampaignIntensityEncoder()),
        ("contact_recency",  ContactRecencyEncoder()),
    ])


def load_data(data_path: str, random_state: int = 42):
    """Load raw Bank Marketing data."""
    df = pd.read_csv(data_path, sep=";")
    df["y"] = (df["y"] == "yes").astype(int)
    X = df.drop(columns=["y"])
    y = df["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    logger.info(f"Loaded: {data_path} | Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def fit_and_log_pipeline(data_path: str = None, random_state: int = 42):
    """
    Fits the feature pipeline on training data and logs as MLflow artifact.
    Decoupled from model training — pipeline version tracked independently.
    """
    # Load data
    if data_path and os.path.exists(data_path):
        X_train, X_test, y_train, y_test = load_data(data_path, random_state)
    else:
        logger.info("No data path — generating synthetic data")
        from src.data.generator import generate_synthetic
        X_train, X_test, y_train, y_test = generate_synthetic(random_state=random_state)

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="feature-pipeline-fit") as run:
        mlflow.set_tag("stage", "feature_engineering")
        mlflow.set_tag("pipeline_version", "v1")
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_features_input", X_train.shape[1])

        # Build and fit pipeline
        pipeline = build_feature_pipeline()
        X_train_transformed = pipeline.fit_transform(X_train)
        X_test_transformed  = pipeline.transform(X_test)

        # Log pipeline metadata
        n_features_out = X_train_transformed.shape[1] if hasattr(X_train_transformed, 'shape') else len(X_train_transformed.columns)
        mlflow.log_param("n_features_output", n_features_out)
        mlflow.log_param("n_transformers", len(pipeline.steps))
        mlflow.log_param("transformers", str([name for name, _ in pipeline.steps]))

        # Log input vs output shapes
        mlflow.log_metric("input_shape_cols",  X_train.shape[1])
        mlflow.log_metric("output_shape_cols", n_features_out)
        mlflow.log_metric("features_added",    n_features_out - X_train.shape[1])

        # Save pipeline as artifact
        os.makedirs("artifacts", exist_ok=True)
        pipeline_path = "artifacts/feature_pipeline.joblib"
        joblib.dump(pipeline, pipeline_path)
        mlflow.log_artifact(pipeline_path)

        # Log fitted pipeline with MLflow
        mlflow.sklearn.log_model(pipeline, ARTIFACT_PATH)

        logger.info(f"Pipeline fitted and logged | Run ID: {run.info.run_id[:8]}...")
        logger.info(f"Input features:  {X_train.shape[1]}")
        logger.info(f"Output features: {n_features_out}")
        logger.info(f"Features added:  {n_features_out - X_train.shape[1]}")

        print("\n" + "="*55)
        print("FEATURE PIPELINE — FIT COMPLETE")
        print("="*55)
        print(f"  Transformers: {len(pipeline.steps)}")
        print(f"  Input cols:   {X_train.shape[1]}")
        print(f"  Output cols:  {n_features_out}")
        print(f"  Added:        {n_features_out - X_train.shape[1]} new features")
        print(f"  Run ID:       {run.info.run_id[:8]}...")
        print(f"  Artifact:     {pipeline_path}")
        print("="*55)

        return pipeline, run.info.run_id, X_train_transformed, X_test_transformed, y_train, y_test


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()
    fit_and_log_pipeline(data_path=args.data_path, random_state=args.random_state)
