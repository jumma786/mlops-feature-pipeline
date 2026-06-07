"""
Model Trainer — Decoupled from Feature Pipeline
================================================
Loads the fitted feature pipeline from MLflow artifacts,
applies it to data, then trains the model separately.

This is the KEY concept of Project 3:
  Feature pipeline version X + Model version Y are tracked independently.
  You can update features without retraining the model and vice versa.
"""

import os
import sys
import time
import logging
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features.pipeline_builder import fit_and_log_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "mlops-feature-pipeline"
MODEL_REGISTRY  = "FeaturePipelineModel"


def train_model(data_path: str = None, random_state: int = 42):
    """
    Step 1: Fit and log feature pipeline
    Step 2: Train model on transformed features
    Both steps logged as separate MLflow runs — fully decoupled.
    """

    # Step 1 — Feature pipeline (separate run)
    logger.info("Step 1: Fitting feature pipeline...")
    pipeline, pipeline_run_id, X_train_t, X_test_t, y_train, y_test = \
        fit_and_log_pipeline(data_path=data_path, random_state=random_state)

    # Convert to DataFrame if needed
    if not isinstance(X_train_t, pd.DataFrame):
        X_train_t = pd.DataFrame(X_train_t)
        X_test_t  = pd.DataFrame(X_test_t)

    # Step 2 — Model training (separate run)
    logger.info("Step 2: Training model on transformed features...")

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="model-training") as run:
        mlflow.set_tag("stage", "model_training")
        mlflow.set_tag("feature_pipeline_run_id", pipeline_run_id[:8])
        mlflow.log_param("feature_pipeline_run_id", pipeline_run_id)
        mlflow.log_param("n_train", len(X_train_t))
        mlflow.log_param("n_features", X_train_t.shape[1])

        # Train
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight="balanced",
            random_state=random_state, n_jobs=-1
        )
        t0 = time.time()
        model.fit(X_train_t, y_train)
        train_time = round(time.time() - t0, 2)

        # Evaluate
        y_pred = model.predict(X_test_t)
        y_prob = model.predict_proba(X_test_t)[:, 1]

        metrics = {
            "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "train_time": train_time,
        }

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print("\n" + "="*55)
        print("MODEL TRAINING — COMPLETE")
        print("="*55)
        print(f"  Feature pipeline run: {pipeline_run_id[:8]}...")
        print(f"  Model run:            {run.info.run_id[:8]}...")
        print(f"  AUC:                  {metrics['roc_auc']:.4f}")
        print(f"  F1:                   {metrics['f1']:.4f}")
        print(f"  Train time:           {train_time}s")
        print("="*55)
        print("\nKEY: Feature pipeline and model are separate MLflow runs.")
        print("     Update features without retraining model — and vice versa.")

        return model, metrics


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()
    train_model(data_path=args.data_path, random_state=args.random_state)
