# src/model.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def compute_forward6_stats(series: pd.Series):
    """
    Compute forward-6-month percent change series: (x[t+6] - x[t]) / x[t] * 100.
    Return (mean_positive_pct, mean_negative_pct, full_forward_series).
    """
    fwd6 = (series.shift(-6) - series) / series * 100.0
    pos_mean = fwd6[fwd6 > 0].mean()
    neg_mean = fwd6[fwd6 < 0].mean()
    pos_mean = pos_mean if pd.notna(pos_mean) else fwd6.mean()
    neg_mean = neg_mean if pd.notna(neg_mean) else fwd6.mean()
    return pos_mean, neg_mean, fwd6

# Train a logistic regression, important thing to note here is that it is trained with a preprocessing pipeline
# StandardScaler is used for numeric features and OneHotEncoder is used for Month

def train_logistic_regression(X: pd.DataFrame, y: pd.Series, random_state=42):
    
    # Identify numeric and categorical columns
    numeric_cols = X.drop(columns=["Month"]).columns.tolist()
    categorical_cols = ["Month"]

    # Preprocessor: scale numeric + one-hot encode Month
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first"), categorical_cols),
        ]
    )

    # Full pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            penalty="l1",
            C=1.0,
            random_state=random_state
        ))
    ])

    # Fit with time-series split (last fold effectively used for training)
    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        pipeline.fit(X_train, y_train)

    return pipeline