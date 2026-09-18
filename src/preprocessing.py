"""
preprocessing.py

Preprocessing for hospital_readmission_dataset.csv, which has columns:
    age, gender, primary_diagnosis, num_procedures, days_in_hospital,
    comorbidity_score, discharge_to, readmitted

- gender, primary_diagnosis, discharge_to are categorical -> one-hot encoded
  (drop_first=True to avoid the dummy-variable trap, which is important
  for a regularized linear model: perfectly collinear dummy columns make
  the L2 penalty split weight arbitrarily between them).
- age, num_procedures, days_in_hospital, comorbidity_score are numeric ->
  standardized (zero mean, unit variance), which matters because L2
  regularization penalizes raw coefficient magnitude, so unscaled features
  would be penalized unevenly.
- readmitted is the binary target.

No sklearn preprocessing classes are used -- implemented directly with numpy/pandas
so the "from scratch" scope covers the full pipeline, not just the model fit step.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

TARGET_COL = "readmitted"
CATEGORICAL_COLS = ["gender", "primary_diagnosis", "discharge_to"]
NUMERIC_COLS = ["age", "num_procedures", "days_in_hospital", "comorbidity_score"]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def one_hot_encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Manual one-hot encoding with drop_first, so behavior is explicit and reproducible."""
    out = df.copy()
    for col in columns:
        dummies = pd.get_dummies(out[col], prefix=col, drop_first=True)
        out = pd.concat([out.drop(columns=[col]), dummies], axis=1)
    return out


class StandardScaler:
    """Minimal from-scratch standardization (zero mean, unit variance)."""

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # guard against constant columns
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    """Stratified train/test split implemented from scratch, so the ~19% base
    readmission rate is preserved in both splits (important with imbalanced
    outcomes -- a non-stratified split can meaningfully shift the minority-class
    rate in the smaller test set)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    n_test_pos = int(len(idx_pos) * test_size)
    n_test_neg = int(len(idx_neg) * test_size)

    test_idx = np.concatenate([idx_pos[:n_test_pos], idx_neg[:n_test_neg]])
    train_idx = np.concatenate([idx_pos[n_test_pos:], idx_neg[n_test_neg:]])
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def build_feature_matrix(df: pd.DataFrame):
    """
    Returns (X, y, feature_names) ready for modeling.
    Numeric columns are left un-scaled here -- scaling happens after the
    train/test split (fit on train only) to avoid leakage.
    """
    df = df.copy()
    y = df[TARGET_COL].values.astype(int)
    df = df.drop(columns=[TARGET_COL])
    df = one_hot_encode(df, CATEGORICAL_COLS)

    # Ensure a consistent, explicit column order
    feature_names = list(df.columns)
    X = df[feature_names].values.astype(float)
    return X, y, feature_names
