"""
Synthetic data generator for feature pipeline testing.
Mirrors UCI Bank Marketing schema.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def generate_synthetic(n_samples: int = 5000, random_state: int = 42):
    """Generate synthetic Bank Marketing data for pipeline testing."""
    np.random.seed(random_state)
    n = n_samples

    df = pd.DataFrame({
        "age":           np.random.randint(18, 95, n),
        "job":           np.random.choice(["admin.","blue-collar","management","retired","technician","unknown"], n),
        "marital":       np.random.choice(["divorced","married","single"], n),
        "education":     np.random.choice(["basic.4y","high.school","university.degree","unknown"], n),
        "default":       np.random.choice(["no","yes","unknown"], n, p=[0.79,0.01,0.20]),
        "housing":       np.random.choice(["no","yes"], n),
        "loan":          np.random.choice(["no","yes"], n, p=[0.82,0.18]),
        "contact":       np.random.choice(["cellular","telephone"], n, p=[0.63,0.37]),
        "month":         np.random.choice(["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], n),
        "day_of_week":   np.random.choice(["mon","tue","wed","thu","fri"], n),
        "duration":      np.random.exponential(250, n).astype(int),
        "campaign":      np.random.randint(1, 15, n),
        "pdays":         np.where(np.random.rand(n) < 0.13, np.random.randint(1,30,n), 999),
        "previous":      np.random.randint(0, 7, n),
        "poutcome":      np.random.choice(["failure","nonexistent","success"], n, p=[0.10,0.86,0.04]),
        "emp.var.rate":  np.random.choice([-1.8,-1.7,1.1,1.4], n),
        "cons.price.idx": np.random.uniform(92.2, 94.8, n).round(3),
        "cons.conf.idx": np.random.uniform(-50.8, -26.9, n).round(1),
        "euribor3m":     np.random.uniform(0.6, 5.1, n).round(3),
        "nr.employed":   np.random.choice([4963.6,5008.7,5099.1,5176.3,5228.1], n),
    })

    y = (np.random.rand(n) < 0.11).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=random_state
    )

    logger.info(f"Generated synthetic data: {df.shape} | positive rate: {y.mean():.1%}")
    return X_train, X_test, y_train, y_test
