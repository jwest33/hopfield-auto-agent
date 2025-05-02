# module_world.py

from __future__ import annotations
import random
import logging
from typing import Tuple, Set
from dataclasses import dataclass, field

import numpy as np

# ───────────────────── TerrainCell Definition ─────────────────────

@dataclass
class TerrainCell:
    # Geometry: slope & surface orientation
    height_vector: Tuple[float, float] = (0.0, 0.0)
    normal_vector: Tuple[float, float] = (0.0, 0.0)

    # Physics properties
    hardness:           float           = 1.0   # resistance to deformation
    strength:           float           = 1.0   # resistance to fracture/removal
    density:            float           = 1.0   # mass per unit volume
    friction:           float           = 1.0   # base friction coefficient
    elasticity:         float           = 0.0   # bounce/rebound factor
    thermal_conductivity: float         = 0.1   # how fast cell temp → ambient
    temperature:        float           = 20.0  # current cell temp (°C)

    # Traversal flags & extras
    passable:           bool            = True
    material:           str             = "dirt"
    local_risk:         float           = 0.0   # passive damage per entry
    tags:               Set[str]        = field(default_factory=set)


# ───────────────────── World Definition ─────────────────────

class World:
    def __init__(self, grid_size: int, day_length: int = 1000):
        logging.info("🌎 Initializing world …")
        self.grid_size = grid_size

        # Create a grid of default TerrainCells
        self.grid: np.ndarray = np.array(
            [[TerrainCell() for _ in range(grid_size)]
             for _ in range(grid_size)],
            dtype=object
        )

        # Mark “home” at center
        cx, cy = grid_size // 2, grid_size // 2
        home = TerrainCell(
            passable=True,
            material="home",
            tags={"home"},
            temperature=20.0
        )
        self.grid[cx, cy] = home
        self.home = (cx, cy)

        # Time & weather tracking
        self.time = 0
        self.day_length = day_length
        self.is_day = True
        self.weather = "clear"
        self._weather_timer = 0

        # Initialize ambient & cell temperatures
        self.ambient_temperature = self._compute_ambient_temperature()
        for x in range(grid_size):
            for y in range(grid_size):
                self.grid[x, y].temperature = self.ambient_temperature

    # ─────────── Core Accessors ───────────

    def cell(self, pos: Tuple[int, int]) -> TerrainCell:
        x, y = pos
        return self.grid[x % self.grid_size, y % self.grid_size]

    def is_passable(self, pos: Tuple[int, int]) -> bool:
        return self.cell(pos).passable

    def traversal_cost(
        self,
        pos: Tuple[int, int],
        direction: Tuple[int, int],
    ) -> float:
        """
        Energy/time cost based on slope, density, friction,
        plus dynamic modifiers for river & weather.
        """
        cell = self.cell(pos)
        dx, dy = direction

        # compute slope effort
        slope = abs(dx * cell.height_vector[0] + dy * cell.height_vector[1])
        # apply friction modifiers
        friction_local = cell.friction * self._weather_friction_modifier()
        if "river" in cell.tags:
            friction_local *= 0.5

        effort = slope * cell.density * friction_local
        return 1.0 + effort

    def hazard_risk(self, pos: Tuple[int, int]) -> float:
        """Risk (pain) from slope steepness and material."""
        cell = self.cell(pos)
        steepness = np.hypot(*cell.height_vector)
        risk = cell.local_risk + steepness * 5.0
        if cell.material in ("stone", "rock"):
            risk *= 1.5
        return risk

    def can_be_broken(self, pos: Tuple[int, int], force: float) -> bool:
        return force >= self.cell(pos).strength

    def can_modify(self, pos: Tuple[int, int], tool_hardness: float) -> bool:
        return tool_hardness >= self.cell(pos).hardness

    # ─────────── Procedural Generators ───────────

    def add_boulder(self, center: Tuple[int, int], radius: int):
        """Create a rounded hill/boulder."""
        cx, cy = center
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    dist = np.hypot(dx, dy)
                    if dist <= radius:
                        height = (radius - dist) / radius
                        slope = (dx * height, dy * height)
                        self.grid[x, y] = TerrainCell(
                            height_vector=slope,
                            normal_vector=(-slope[1], slope[0]),
                            material="stone",
                            passable=True,
                            hardness=7.0,
                            strength=8.0,
                            density=2.5,
                            friction=1.2 + height,
                            elasticity=0.1 * height,
                            thermal_conductivity=0.3,
                            temperature=self.ambient_temperature,
                            local_risk=0.05 * height,
                            tags={"boulder"},
                        )

    def add_ridge(
        self,
        start: Tuple[int, int],
        length: int,
        direction: Tuple[int, int]
    ):
        """Create an impassable cliff/ridge."""
        x, y = start
        dx, dy = direction
        for _ in range(length):
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                slope = (dx * 5.0, dy * 5.0)
                self.grid[x, y] = TerrainCell(
                    height_vector=slope,
                    normal_vector=(-dy, dx),
                    material="rock",
                    passable=False,
                    hardness=9.0,
                    strength=10.0,
                    density=2.7,
                    friction=1.0,
                    elasticity=0.0,
                    thermal_conductivity=0.2,
                    temperature=self.ambient_temperature,
                    tags={"cliff", "ridge"},
                )
            x += dx
            y += dy

    def add_river(
        self,
        origin: Tuple[int, int],
        direction: Tuple[int, int],
        length: int
    ):
        """Draw a river with flow but higher base friction & risk."""
        x, y = origin
        dx, dy = direction
        for _ in range(length):
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self.grid[x, y] = TerrainCell(
                    height_vector=(0.0, 0.0),
                    normal_vector=(0.0, 0.0),
                    material="water",
                    passable=True,
                    hardness=0.5,
                    strength=0.5,
                    density=1.0,
                    friction=2.0,
                    elasticity=0.0,
                    thermal_conductivity=0.15,
                    temperature=self.ambient_temperature,
                    local_risk=0.2,
                    tags={"river"},
                )
            x += dx
            y += dy

    def add_forest(self, center: Tuple[int, int], radius: int):
        """Cluster of impassable trees with tangled vectors."""
        cx, cy = center
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = cx + dx, cy + dy
                if (0 <= x < self.grid_size and 0 <= y < self.grid_size
                        and np.hypot(dx, dy) <= radius):
                    angle = random.uniform(0, 2 * np.pi)
                    vec = (np.cos(angle), np.sin(angle))
                    self.grid[x, y] = TerrainCell(
                        height_vector=vec,
                        normal_vector=(-vec[1], vec[0]),
                        material="wood",
                        passable=False,
                        hardness=2.0,
                        strength=4.0,
                        density=0.9,
                        friction=1.5,
                        elasticity=0.0,
                        thermal_conductivity=0.25,
                        temperature=self.ambient_temperature,
                        tags={"forest", "tree"},
                    )

    # ─────────── Internal Helpers ───────────

    def _weather_friction_modifier(self) -> float:
        """Global friction adjustment by weather."""
        return {
            "clear": 1.0,
            "rain": 0.8,
            "storm": 0.6
        }[self.weather]

    def _compute_ambient_temperature(self) -> float:
        """
        Continuous diurnal cycle between min/max temps.
        Uses a sine wave phased so peak at midday.
        """
        min_t, max_t = 10.0, 30.0
        # fraction of day [0,1)
        frac = (self.time / self.day_length) % 1.0
        # shift so peak at frac=0.5 (midday)
        theta = 2 * np.pi * (frac - 0.25)
        return min_t + (max_t - min_t) * (0.5 * (1 + np.sin(theta)))

    # ─────────── World Update ───────────

    def step(self):
        """
        Advance time, toggle day/night, update weather,
        and relax cell temperatures toward ambient.
        """
        # Time & Day/Night
        self.time = (self.time + 1) % self.day_length
        self.is_day = (self.time < self.day_length / 2)

        # Weather cycle
        if self._weather_timer <= 0:
            self.weather = random.choice(["clear", "rain", "storm"])
            self._weather_timer = random.randint(200, 600)
        else:
            self._weather_timer -= 1

        # Ambient temperature update
        self.ambient_temperature = self._compute_ambient_temperature()

        # Relax each cell’s temperature toward ambient
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell = self.grid[x, y]
                delta = self.ambient_temperature - cell.temperature
                cell.temperature += cell.thermal_conductivity * delta
