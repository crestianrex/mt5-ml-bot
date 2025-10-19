# src/linear_thompson.py
from __future__ import annotations
import numpy as np
from typing import List

class LinearThompson:
    """
    Lightweight contextual Thompson Sampling.
    Independent Bayesian linear model per arm:
      reward ~ N(w^T x, noise_var)
      prior w ~ N(0, (1/lambda_prior) I)
    Small, fast (dim small).
    """
    def __init__(self, num_arms: int, dim: int, lambda_prior: float = 1.0, noise_var: float = 1.0):
        self.num_arms = int(num_arms)
        self.dim = int(dim)
        self.lambda_prior = float(lambda_prior)
        self.noise_var = float(noise_var)

        self.A: List[np.ndarray] = [self.lambda_prior * np.eye(self.dim) for _ in range(self.num_arms)]
        self.b: List[np.ndarray] = [np.zeros(self.dim) for _ in range(self.num_arms)]
        self.invA: List[np.ndarray] = [np.linalg.inv(a) for a in self.A]

    def sample_arm(self, x: np.ndarray) -> int:
        x = x.reshape(-1)
        vals = np.empty(self.num_arms, dtype=float)
        for i in range(self.num_arms):
            mu = self.invA[i].dot(self.b[i])                  # posterior mean
            cov = self.noise_var * self.invA[i]               # posterior covariance
            try:
                w_samp = np.random.multivariate_normal(mu, cov)
            except Exception:
                # numeric fallback
                w_samp = mu + np.random.randn(self.dim) * np.sqrt(np.diag(cov) + 1e-6)
            vals[i] = float(w_samp.dot(x))
        return int(np.argmax(vals))

    def update(self, arm: int, x: np.ndarray, reward: float):
        x = x.reshape(-1)
        self.A[arm] += np.outer(x, x) / max(1e-9, self.noise_var)
        self.b[arm] += x * (reward / max(1e-9, self.noise_var))
        # Recompute inverse (dim is small, so direct inverse ok)
        try:
            self.invA[arm] = np.linalg.inv(self.A[arm])
        except np.linalg.LinAlgError:
            # Regularize slightly and invert
            self.invA[arm] = np.linalg.inv(self.A[arm] + 1e-6 * np.eye(self.dim))

    def get_state(self) -> dict:
        return {
            "num_arms": self.num_arms,
            "dim": self.dim,
            "lambda_prior": self.lambda_prior,
            "noise_var": self.noise_var,
            "A": [a.tolist() for a in self.A],
            "b": [b.tolist() for b in self.b],
        }

    @classmethod
    def from_state(cls, state: dict) -> "LinearThompson":
        inst = cls(int(state["num_arms"]), int(state["dim"]), float(state.get("lambda_prior", 1.0)), float(state.get("noise_var", 1.0)))
        inst.A = [np.array(a) for a in state.get("A", inst.A)]
        inst.b = [np.array(b) for b in state.get("b", inst.b)]
        inst.invA = [np.linalg.inv(a) for a in inst.A]
        return inst
