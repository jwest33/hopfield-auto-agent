from __future__ import annotations
import time
import numpy as np

# ───────────────────── configuration ─────────────────────
BETA = 2.0
SURPRISE_SCALE = 10

# ───────────────────── Hopfield ─────────────────────
class Hopfield:
    def __init__(self, dim: int, cap: int):
        self.dim, self.cap, self.beta = dim, cap, BETA
        self.M = np.empty((0, dim))
        self.t: list[float] = []

    # ——— internal helpers ———
    def _evict(self):
        while len(self.t) > self.cap:
            idx = int(np.argmin(self.t))
            self.M = np.delete(self.M, idx, 0)
            self.t.pop(idx)

    # ——— public API ———
    def store(self, v: np.ndarray):
        if v.size != self.dim:
            return
        self.M = np.vstack([self.M, v])
        self.t.append(time.time())
        self._evict()

    def recall(self, v: np.ndarray, it: int = 3) -> np.ndarray:
        if self.M.size == 0:
            return v.copy()
        y = v.copy()
        for _ in range(it):
            logits = self.M @ y - (self.M @ y).max()
            p = np.exp(logits)
            s = p.sum()
            if not np.isfinite(s) or s == 0:
                return v.copy()
            y = (p / s) @ self.M
        return y

    def surprise(self, v: np.ndarray) -> float:
        return np.linalg.norm(self.recall(v) - v) * SURPRISE_SCALE if self.M.size else 5.0
