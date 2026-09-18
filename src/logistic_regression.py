"""
logistic_regression.py

A from-scratch implementation of L2-regularized (ridge) logistic regression,
trained via batch gradient descent on the log loss:

    J(w) = -(1/n) * sum[ y*log(p) + (1-y)*log(1-p) ]  +  (lambda / (2n)) * ||w||^2

Only numpy is used for the math -- no scikit-learn model classes.
This mirrors what "logistic regression with L2 regularization" means
mathematically, and makes the regularization / optimization explicit
rather than hiding it inside a library call.
"""

from __future__ import annotations
import numpy as np


class LogisticRegressionL2:
    """
    L2-regularized logistic regression trained with batch gradient descent.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent.
    n_iterations : int
        Number of full-batch gradient descent steps.
    l2_lambda : float
        L2 regularization strength. lambda=0 recovers plain logistic regression.
        The bias term is NOT regularized (standard convention).
    class_weight : dict or None
        Optional {0: w0, 1: w1} sample weighting to counteract class imbalance,
        e.g. {0: 1.0, 1: 4.0} to upweight the minority (readmitted) class.
    fit_intercept : bool
        Whether to learn a bias term.
    tol : float
        Stop early if the change in loss between iterations drops below this.
    verbose : bool
        Print loss every 100 iterations.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_iterations: int = 5000,
        l2_lambda: float = 1.0,
        class_weight: dict | None = None,
        fit_intercept: bool = True,
        tol: float = 1e-7,
        verbose: bool = False,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2_lambda = l2_lambda
        self.class_weight = class_weight
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.verbose = verbose

        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.loss_history_: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[~pos])
        out[~pos] = exp_z / (1.0 + exp_z)
        return out

    def _sample_weights(self, y: np.ndarray) -> np.ndarray:
        if self.class_weight is None:
            return np.ones_like(y, dtype=float)
        w = np.where(y == 1, self.class_weight.get(1, 1.0), self.class_weight.get(0, 1.0))
        return w.astype(float)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionL2":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape

        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        sw = self._sample_weights(y)
        sw_sum = sw.sum()

        prev_loss = np.inf
        self.loss_history_ = []

        for i in range(self.n_iterations):
            z = X @ self.weights_ + self.bias_
            p = self._sigmoid(z)
            error = p - y  # dL/dz for each sample

            # Weighted gradients
            grad_w = (X.T @ (sw * error)) / sw_sum + (self.l2_lambda / sw_sum) * self.weights_
            grad_b = np.sum(sw * error) / sw_sum if self.fit_intercept else 0.0

            self.weights_ -= self.learning_rate * grad_w
            if self.fit_intercept:
                self.bias_ -= self.learning_rate * grad_b

            # Track loss (weighted log loss + L2 penalty) every 50 steps
            if i % 50 == 0 or i == self.n_iterations - 1:
                eps = 1e-12
                log_loss = -np.sum(
                    sw * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
                ) / sw_sum
                penalty = (self.l2_lambda / (2 * sw_sum)) * np.sum(self.weights_**2)
                loss = log_loss + penalty
                self.loss_history_.append(loss)

                if self.verbose and i % 500 == 0:
                    print(f"  iter {i:5d}  loss = {loss:.5f}")

                if abs(prev_loss - loss) < self.tol:
                    break
                prev_loss = loss

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        z = X @ self.weights_ + self.bias_
        return self._sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def coef_summary(self, feature_names: list[str]) -> list[tuple[str, float, float]]:
        """Return (feature_name, coefficient, odds_ratio) sorted by |coefficient|, descending."""
        pairs = list(zip(feature_names, self.weights_))
        pairs.sort(key=lambda t: abs(t[1]), reverse=True)
        return [(name, coef, float(np.exp(coef))) for name, coef in pairs]
