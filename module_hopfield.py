from __future__ import annotations
import time
import numpy as np

# ───────────────────── configuration ─────────────────────
BETA = 2.0
SURPRISE_SCALE = 10

# ───────────────────── Hopfield ─────────────────────
class Hopfield:
    """
    Continuous (modern) Hopfield network using softmax-based attention over stored patterns.
        References:
    - Ramsauer et al. "Hopfield Networks is All You Need" (2020)
    """
    def __init__(self, dim: int, cap: int, beta: float = 2.0, max_iter: int = 3):
        self.dim = dim
        self.cap = cap
        self.beta = beta
        self.max_iter = max_iter
        # Memory matrix: each row is a stored pattern
        self.M = np.empty((0, dim))

    def store(self, v: np.ndarray):
        """Store a new pattern, evicting the oldest if over capacity."""
        if v.size != self.dim:
            return
        # Append new pattern
        self.M = np.vstack([self.M, v.reshape(1, -1)])
        # Evict oldest beyond capacity
        if self.M.shape[0] > self.cap:
            self.M = self.M[-self.cap :, :]

    def recall(self, v: np.ndarray) -> np.ndarray:
        """Retrieve attractor state via iterative attention."""
        if self.M.shape[0] == 0:
            return v.copy()
        y = v.reshape(1, -1)  # shape (1, dim)
        for _ in range(self.max_iter):
            # Attention logits: (1, n_patterns)
            logits = self.beta * (y @ self.M.T)
            # Stability
            logits = logits - np.max(logits, axis=1, keepdims=True)
            weights = np.exp(logits)
            weights = weights / np.sum(weights, axis=1, keepdims=True)
            # Weighted sum over memory
            y = weights @ self.M  # shape (1, dim)
        return y.flatten()

    def surprise(self, v: np.ndarray) -> float:
        """Compute surprise as the Euclidean distance to the recalled pattern."""
        rec = self.recall(v)
        return np.linalg.norm(rec - v)
