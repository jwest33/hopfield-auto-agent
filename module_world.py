from __future__ import annotations
import random, logging, os
from typing import Tuple
import numpy as np

# ───────────────────── World ─────────────────────
class World:
    # Path to persist world state
    STATE_FILE = "world_state.npz"

    def __init__(self, grid_size: int):
        logging.info("Initializing world...")
        self.grid_size = grid_size

        # If a saved world exists, load it
        if os.path.exists(self.STATE_FILE):
            data = np.load(self.STATE_FILE, allow_pickle=True)
            self.grid = data["grid"]
            self.home = tuple(data["home"])
            logging.info("Loaded world from saved state.")
        else:
            # Create fresh world and save
            self.grid = np.full((grid_size, grid_size), "empty", object)
            self.home = (self.grid_size // 2, self.grid_size // 2)
            self.grid[self.home] = "home"

            # Use a fixed RNG for deterministic placement
            rnd = random.Random(42)
            for _ in range(80):
                self._rand("food", rnd)
            for _ in range(70):
                self._rand("hazard", rnd)

            # Persist the generated world
            np.savez_compressed(
                self.STATE_FILE,
                grid=self.grid,
                home=np.array(self.home)
            )
            logging.info("Generated and saved new world state.")

    def _rand(self, label: str, rnd: random.Random):
        """Place a label in a random empty cell using provided RNG."""
        while True:
            x = rnd.randrange(self.grid_size)
            y = rnd.randrange(self.grid_size)
            if self.grid[x, y] == "empty":
                self.grid[x, y] = label
                break

    def cell(self, pos: Tuple[int, int]) -> str:
        x, y = pos
        return self.grid[x % self.grid_size, y % self.grid_size]

    def remove_food(self, pos: Tuple[int, int]):
        # Remove food and update persistent state
        self.grid[pos] = "empty"
        self._save_state()

    def nearest_food_distance(self, pos: Tuple[int, int]) -> int:
        fx, fy = np.where(self.grid == "food")
        if fx.size == 0:
            return self.grid_size
        dists = np.abs(fx - pos[0]) + np.abs(fy - pos[1])
        return int(dists.min())

    def _save_state(self):
        # Persist the current grid and home location
        np.savez_compressed(
            self.STATE_FILE,
            grid=self.grid,
            home=np.array(self.home)
        )
