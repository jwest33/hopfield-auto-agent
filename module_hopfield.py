from __future__ import annotations
import numpy as np

class Hopfield:
    """
    Continuous modern Hopfield network that can dynamically adapt to the dimension
    of the stored patterns (e.g., full paths + outcome vectors).
    """
    def __init__(self, cap: int, beta: float = 2.0, max_iter: int = 3):
        self.cap = cap
        self.beta = beta
        self.max_iter = max_iter
        self.dim: int | None = None
        # Memory matrix: each row is a stored pattern
        self.M: np.ndarray = np.empty((0, 0))

    def store(self, v: np.ndarray):
        # Flatten and determine dimension on first store
        vec = v.flatten()
        if self.dim is None:
            self.dim = vec.size
            self.M = vec.reshape(1, -1)
        else:
            # If incoming vector smaller/larger, pad or truncate
            if vec.size < self.dim:
                pad = np.zeros(self.dim - vec.size)
                vec = np.concatenate([vec, pad])
            elif vec.size > self.dim:
                vec = vec[: self.dim]
            self.M = np.vstack([self.M, vec.reshape(1, -1)])
        # Evict oldest beyond capacity
        if self.M.shape[0] > self.cap:
            self.M = self.M[-self.cap :]

    def recall(self, v: np.ndarray) -> np.ndarray:
        if self.M.shape[0] == 0:
            return v.flatten().copy()
        y = v.flatten().reshape(1, -1)
        for _ in range(self.max_iter):
            # dot product attention
            logits = self.beta * (y @ self.M.T)
            logits = logits - np.max(logits, axis=1, keepdims=True)
            weights = np.exp(logits)
            weights = weights / np.sum(weights, axis=1, keepdims=True)
            y = weights @ self.M
        return y.flatten()

    def surprise(self, v: np.ndarray) -> float:
        rec = self.recall(v)
        return np.linalg.norm(rec - v.flatten())
