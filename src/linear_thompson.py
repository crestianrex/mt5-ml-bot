# src/linear_thompson.py
from __future__ import annotations
import numpy as np
from typing import List
import logging
from collections import deque # NEW

logger = logging.getLogger(__name__)

class LinearThompson:
    """
    Lightweight contextual Thompson Sampling.
    Independent Bayesian linear model per arm:
      reward ~ N(w^T x, noise_var)
      prior w ~ N(0, (1/lambda_prior) I)
    Small, fast (dim small).
    """
    def __init__(self, num_arms: int, dim: int, lambda_prior: float = 1.0, noise_var: float = 1.0, dynamic_noise_var_enabled: bool = False, noise_var_window_size: int = 50, min_noise_var: float = 1e-6, dynamic_uncertainty_risk_scaling_enabled: bool = False, uncertainty_risk_factor: float = 0.5, uncertainty_threshold: float = 0.1):
        if not isinstance(num_arms, int) or num_arms <= 0:
            raise ValueError("num_arms must be a positive integer")
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if not isinstance(lambda_prior, (int, float)) or lambda_prior <= 0:
            raise ValueError("lambda_prior must be a positive float")
        if not isinstance(noise_var, (int, float)) or noise_var <= 0:
            raise ValueError("noise_var must be a positive float")
        if not isinstance(noise_var_window_size, int) or noise_var_window_size <= 0:
            raise ValueError("noise_var_window_size must be a positive integer")
        if not isinstance(min_noise_var, (int, float)) or min_noise_var <= 0:
            raise ValueError("min_noise_var must be a positive float")
        if not isinstance(uncertainty_risk_factor, (int, float)) or uncertainty_risk_factor <= 0:
            raise ValueError("uncertainty_risk_factor must be a positive float")
        if not isinstance(uncertainty_threshold, (int, float)) or uncertainty_threshold <= 0:
            raise ValueError("uncertainty_threshold must be a positive float")

        self.num_arms = num_arms
        self.dim = dim
        self.lambda_prior = float(lambda_prior)
        self.initial_noise_var = float(noise_var) # Store initial noise_var
        self.noise_var = float(noise_var) # Current noise_var, can be dynamic

        self.dynamic_noise_var_enabled = dynamic_noise_var_enabled
        self.noise_var_window_size = noise_var_window_size
        self.min_noise_var = min_noise_var

        self.dynamic_uncertainty_risk_scaling_enabled = dynamic_uncertainty_risk_scaling_enabled
        self.uncertainty_risk_factor = uncertainty_risk_factor
        self.uncertainty_threshold = uncertainty_threshold

        # Deques to store residuals for dynamic noise_var estimation
        self.residuals: List[deque] = [deque(maxlen=self.noise_var_window_size) for _ in range(self.num_arms)]

        self.A: List[np.ndarray] = [self.lambda_prior * np.eye(self.dim) for _ in range(self.num_arms)]  # Shape: (dim, dim)
        self.b: List[np.ndarray] = [np.zeros(self.dim) for _ in range(self.num_arms)]  # Shape: (dim,)
        self.invA: List[np.ndarray] = [np.linalg.inv(a) for a in self.A]  # Shape: (dim, dim)

    def sample_arm(self, x: np.ndarray) -> int:
        x = x.reshape(-1)  # Ensure x is 1D, shape (dim,)
        vals = np.empty(self.num_arms, dtype=float)
        for i in range(self.num_arms):
            mu = self.invA[i].dot(self.b[i])                  # posterior mean, shape (dim,)
            cov = self.noise_var * self.invA[i]               # posterior covariance, shape (dim, dim)

            # Ensure covariance matrix is positive semi-definite
            if not np.all(np.linalg.eigvals(cov) >= 0):
                logger.warning(f"Covariance matrix for arm {i} is not positive semi-definite. Falling back to diagonal covariance sampling.")
                # Fallback to diagonal covariance sampling
                w_samp = mu + np.random.randn(self.dim) * np.sqrt(np.diag(cov).clip(min=1e-6))
            else:
                try:
                    w_samp = np.random.multivariate_normal(mu, cov)
                except np.linalg.LinAlgError:
                    logger.warning(f"Multivariate normal sampling failed for arm {i}. Falling back to diagonal covariance sampling.")
                    # numeric fallback
                    w_samp = mu + np.random.randn(self.dim) * np.sqrt(np.diag(cov).clip(min=1e-6))
                except Exception as e:
                    logger.warning(f"An unexpected error occurred during multivariate normal sampling for arm {i}: {e}. Falling back to diagonal covariance sampling.")
                    w_samp = mu + np.random.randn(self.dim) * np.sqrt(np.diag(cov).clip(min=1e-6))
            vals[i] = float(w_samp.dot(x))
        return int(np.argmax(vals))

    def update(self, arm: int, x: np.ndarray, reward: float, decay: float = 1.0):
        x = x.reshape(-1)
        x_outer_x = np.outer(x, x)
        noise_var_inv = 1.0 / max(1e-9, self.noise_var)

        # Apply decay to previous A and b
        self.A[arm] *= decay
        self.b[arm] *= decay

        # Update A and b
        self.A[arm] += x_outer_x * noise_var_inv
        self.b[arm] += x * (reward * noise_var_inv)

        # Calculate predicted reward for residual
        w_mean = self.invA[arm].dot(self.b[arm])
        predicted_reward = w_mean.dot(x)
        residual = reward - predicted_reward
        self.residuals[arm].append(residual)

        # Dynamically update noise_var if enabled
        if self.dynamic_noise_var_enabled and len(self.residuals[arm]) >= self.noise_var_window_size:
            estimated_noise_var = np.var(list(self.residuals[arm]))
            self.noise_var = max(self.min_noise_var, estimated_noise_var)
            # logger.debug(f"Arm {arm}: Dynamic noise_var updated to {self.noise_var:.6f}")

        # Update invA using Sherman-Morrison formula for rank-1 update
        # (A + uv^T)^-1 = A^-1 - (A^-1 u v^T A^-1) / (1 + v^T A^-1 u)
        # Here, u = x, v = x * noise_var_inv
        invA_old = self.invA[arm]
        u = x
        v = x * noise_var_inv

        try:
            numerator = invA_old @ np.outer(u, v) @ invA_old
            denominator = 1.0 + v.T @ invA_old @ u
            self.invA[arm] = invA_old - numerator / denominator
        except np.linalg.LinAlgError:
            # Fallback to direct inverse with regularization if Sherman-Morrison fails
            self.invA[arm] = np.linalg.inv(self.A[arm] + 1e-6 * np.eye(self.dim))

    def get_state(self) -> dict:
        return {
            "num_arms": self.num_arms,
            "dim": self.dim,
            "lambda_prior": self.lambda_prior,
            "noise_var": self.noise_var,
            "dynamic_noise_var_enabled": self.dynamic_noise_var_enabled,
            "noise_var_window_size": self.noise_var_window_size,
            "min_noise_var": self.min_noise_var,
            "dynamic_uncertainty_risk_scaling_enabled": getattr(self, "dynamic_uncertainty_risk_scaling_enabled", False),
            "uncertainty_risk_factor": getattr(self, "uncertainty_risk_factor", 0.5),
            "uncertainty_threshold": getattr(self, "uncertainty_threshold", 0.1),
            "A": [a.tolist() for a in self.A],
            "b": [b.tolist() for b in self.b],
            "residuals": [list(r) for r in self.residuals] # Save residuals for state reconstruction
        }

    @classmethod
    def from_state(cls, state: dict) -> "LinearThompson":
        inst = cls(
            int(state["num_arms"]),
            int(state["dim"]),
            float(state.get("lambda_prior", 1.0)),
            float(state.get("noise_var", 1.0)),
            bool(state.get("dynamic_noise_var_enabled", False)),
            int(state.get("noise_var_window_size", 50)),
            float(state.get("min_noise_var", 1e-6)),
            bool(state.get("dynamic_uncertainty_risk_scaling_enabled", False)),
            float(state.get("uncertainty_risk_factor", 0.5)),
            float(state.get("uncertainty_threshold", 0.1))
        )
        inst.A = [np.array(a) for a in state.get("A", inst.A)]
        inst.b = [np.array(b) for b in state.get("b", inst.b)]
        inst.invA = [np.linalg.inv(a) for a in inst.A]
        # Reconstruct residuals deque if dynamic noise_var is enabled
        if inst.dynamic_noise_var_enabled:
            for i, res_list in enumerate(state.get("residuals", [])):
                inst.residuals[i] = deque(res_list, maxlen=inst.noise_var_window_size)
        return inst
