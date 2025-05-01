from __future__ import annotations
import random, logging
from typing import Tuple
import numpy as np


# ───────────────────── World ─────────────────────
class World:
    def __init__(self, grid_size: int):
        logging.info("Creating new world grid …")
        self.grid = np.full((grid_size, grid_size), "empty", object)
        self.grid_size = grid_size
        self.home = (self.grid_size // 2, self.grid_size // 2)
        self.grid[self.home] = "home"
        for _ in range(80):
            self._rand("food")
        for _ in range(70):
            self._rand("hazard")

    def _rand(self, label: str):
        while True:
            x, y = random.randrange(self.grid_size), random.randrange(self.grid_size)
            if self.grid[x, y] == "empty":
                self.grid[x, y] = label
                break

    def cell(self, pos: Tuple[int, int]) -> str:
        x, y = pos
        return self.grid[x % self.grid_size, y % self.grid_size]

    def remove_food(self, pos: Tuple[int, int]):
        self.grid[pos] = "empty"

    def nearest_food_distance(self, pos: Tuple[int, int]) -> int:
        fx, fy = np.where(self.grid == "food")
        if fx.size == 0:
            return self.grid_size
        dists = np.abs(fx - pos[0]) + np.abs(fy - pos[1])
        return int(dists.min())

    def replenish_resources(self, rate=0.01):
        """Occasionally replenish food in the world"""
        # Count current food
        food_count = (self.grid == "food").sum()
        
        # Replenish if below threshold
        if food_count < 40 and random.random() < rate:
            self._rand("food")
    
    def step(self):
        """Update world state for one tick"""
        # Replenish resources
        self.replenish_resources()
