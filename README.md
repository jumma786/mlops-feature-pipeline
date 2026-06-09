# ⚙️ Feature Engineering Pipeline as Versioned Artifact

![CI](https://github.com/jumma786/mlops-feature-pipeline/actions/workflows/feature_pipeline.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![MLflow](https://img.shields.io/badge/MLflow-3.13-orange)
![Sklearn](https://img.shields.io/badge/Scikit--learn-1.5-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Part of the [MLOps Portfolio Series](https://github.com/jumma786/mlops-portfolio)** — Project 3 of 10  
> Feature engineering pipeline packaged as versioned sklearn transformers, logged as MLflow artifacts, and decoupled from the model — either can be retrained independently.

---

## 📂 Project Resources

| Resource | Link |
|---|---|
| ⚙️ Transformers | [src/features/transformers.py](src/features/transformers.py) |
| 🔧 Pipeline Builder | [src/features/pipeline_builder.py](src/features/pipeline_builder.py) |
| 🤖 Model Trainer | [src/train.py](src/train.py) |
| 📦 Data Generator | [src/data/generator.py](src/data/generator.py) |
| 🧪 Unit Tests | [tests/test_feature_pipeline.py](tests/test_feature_pipeline.py) |
| 🤖 CI/CD Workflow | [.github/workflows/feature_pipeline.yml](.github/workflows/feature_pipeline.yml) |
| 📋 Requirements | [requirements.txt](requirements.txt) |

---

## 🎯 What This Project Does

Solves the "feature/model coupling" problem — in most pipelines, changing features requires retraining the model:

1. **Packages 5 custom sklearn transformers** — each leakage-safe, unit-tested, and independently versioned
2. **Logs the fitted pipeline as an MLflow artifact** — separate from the model
3. **Decouples feature engineering from model training** — update features without retraining model
4. **Tracks feature versions independently** — feature pipeline run ID linked to model run ID

---

## 🔧 The 5 Transformers

| Transformer | Input | Output | Purpose |
|---|---|---|---|
| `DurationDropper` | `duration` col | Removed | Drops leakage feature |
| `CategoricalEncoder` | Object columns | Integer encoded | Label encodes all categoricals |
| `EconomicFeatureEngineer` | `euribor3m`, `emp.var.rate` | `euribor_emp_interaction` | Economic downturn signal |
| `CampaignIntensityEncoder` | `campaign` | `campaign_intensity` | Customer fatigue signal |
| `ContactRecencyEncoder` | `pdays` | `contact_recency` | Previous contact recency |

**Input:** 20 features → **Output:** 22 features (2 new engineered features)

---

## 🏗️ Architecture

```
mlops-feature-pipeline/
├── src/
│   ├── features/
│   │   ├── transformers.py       # 5 custom sklearn transformers
│   │   └── pipeline_builder.py  # Assembles + logs pipeline as MLflow artifact
│   ├── data/
│   │   └── generator.py         # Synthetic data generator
│   └── train.py                 # Model trainer — loads pipeline, trains separately
├── tests/
│   └── test_feature_pipeline.py # 17 unit tests
├── .github/
│   └── workflows/
│       └── feature_pipeline.yml # CI: test → fit pipeline → train model
├── requirements.txt
└── Makefile
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/jumma786/mlops-feature-pipeline.git
cd mlops-feature-pipeline
pip install -r requirements.txt

# Run tests
make test

# Fit pipeline + train model (2 separate MLflow runs)
make pipeline

# View in MLflow UI
make mlflow-ui
# → Open http://localhost:5000
```

---

## 📈 Results

| Step | MLflow Run | Features In | Features Out | Added |
|---|---|---|---|---|
| Feature Pipeline | `feature-pipeline-fit` | 20 | 22 | 2 |
| Model Training | `model-training` | 22 | — | — |

Both runs linked via `feature_pipeline_run_id` tag — full lineage tracked.

---

## 🔑 Key Concept — Decoupling

```
Traditional approach:         MLOps approach (this project):
─────────────────────         ──────────────────────────────
features + model = 1 blob     feature pipeline = artifact v1
change features → retrain     model = artifact v2
everything                    change features → new pipeline v2
                              model unchanged if performance OK
```

---

## 🔗 MLOps Portfolio Series

| # | Project | Repo | Status |
|---|---|---|---|
| 1 | Multi-Model Tournament | [mlops-model-tournament](https://github.com/jumma786/mlops-model-tournament) | ✅ |
| 2 | Scheduled Retraining | [mlops-retraining-pipeline](https://github.com/jumma786/mlops-retraining-pipeline) | ✅ |
| **3** | **Feature Engineering** | [mlops-feature-pipeline](https://github.com/jumma786/mlops-feature-pipeline) | ✅ This repo |
| 4 | Hyperparameter Tuning | [mlops-hyperparameter-tuning](https://github.com/jumma786/mlops-hyperparameter-tuning) | ✅ |
| 5 | Model Serving | [mlops-model-serving](https://github.com/jumma786/mlops-model-serving) | ✅ |
| 6 | Feature Store | [mlops-feature-store](https://github.com/jumma786/mlops-feature-store) | ✅ |
| 7 | Model Monitoring | [mlops-model-monitoring](https://github.com/jumma786/mlops-model-monitoring) | ✅ |
| 8 | A/B Testing | [mlops-ab-testing](https://github.com/jumma786/mlops-ab-testing) | ✅ |
| 9 | Airflow Pipeline | [mlops-airflow-pipeline](https://github.com/jumma786/mlops-airflow-pipeline) | ✅ |
| 10 | Kubernetes Platform | [mlops-k8s-platform](https://github.com/jumma786/mlops-k8s-platform) | ✅ |

---

## 📝 Key MLOps Concepts Demonstrated

- **Versioned feature pipelines** — fitted pipeline logged as MLflow artifact
- **Decoupling** — feature pipeline and model tracked as separate runs
- **Leakage discipline** — `duration` dropped in first transformer step
- **Custom sklearn transformers** — `BaseEstimator` + `TransformerMixin` pattern
- **Unit testing transformers** — 17 tests covering each transformer individually
- **CI/CD** — GitHub Actions fit → train → artifact upload

---

## 👤 Author

**Jumma Mohammad Teli** — Data Analyst & ML Engineer  
📍 Birmingham, UK  
📧 [jummamohammad477@gmail.com](mailto:jummamohammad477@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/jumma-mohammad) | [GitHub](https://github.com/jumma786)

---

*Project 3 of 10 — MLOps Portfolio Series. Builds on Project 1 (Model Tournament) by versioning the feature engineering step separately from the model.*
