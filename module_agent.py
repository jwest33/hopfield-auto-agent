from __future__ import annotations
import itertools, random, time, logging, os, sys
from typing import Tuple, Dict, Any, List
import numpy as np
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from module_hopfield import Hopfield
from module_new_world import World, TerrainCell  # Updated import
from module_coms import NeuralCommunicationSystem

# ───────────────────── configuration ─────────────────────
GRID = 40  # Updated to match module_new_world's default
OBS_DIM = GRID * GRID * 8  # Expanded for more terrain features
SEQ_LEN, SEQ_DIM = 5, OBS_DIM * 5
CAP_L0, CAP_L1 = 800, 1200

MAX_E, MAX_H, MAX_P = 100, 100, 100
MOVE_COST, CARRY_COST = 1, 1
FOOD_E, FOOD_S = 40, 40
PAIN_HIT, PAIN_DECAY = 25, 1
HOME_ENERGY_RECOVERY = 5  # Energy recovery at home
HOME_HUNGER_RECOVERY = 5  # Hunger reduction at home

BETA = 2.0
SURPRISE_SCALE = 10
HUNGER_W, PAIN_W = 1.5, 2.0
CARRY_HOME_W = 3.0
HOME_DIST_W = 2.0
ENERGY_LOW_FRAC = 0.3
EXPLORATION_BONUS = 5.0
REST_PENALTY = 0.2
TICK_SEC = 0.15
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPERIENCE_DECAY = 0.999

STATE_FILE = "agent_state.npz"

# Updated to use material types from the new world system
CELL_TYPES = np.array(["home", "food", "dirt", "stone", "rock", "water", "wood"]).reshape(-1, 1)
ENC = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(CELL_TYPES)

# ───────────────────── Helper Functions ─────────────────────
def is_food_cell(cell: TerrainCell) -> bool:
    """Helper function to determine if a cell contains food"""
    return (cell.material == "food" or "food" in cell.tags)

# ───────────────────── Agent ─────────────────────
class Agent:
    MOV = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1), "REST": (0, 0)}

    def __init__(self, world, agent_id=None, learning_rate=LEARNING_RATE):
        self.w = world
        self.pos = list(world.home)
        self.energy, self.hunger, self.pain = MAX_E, 0, 0
        self.carrying, self.store = False, 0
        self.rest_streak = 0
        self.mem0 = Hopfield(OBS_DIM, CAP_L0)
        self.mem1 = Hopfield(SEQ_DIM, CAP_L1)
        self.trace: list[np.ndarray] = []
        
        # Initialize experience memory
        self.cell_experience = {
            material: {"reward": 0, "visits": 0, "last_visit": 0}
            for material in ["home", "food", "dirt", "stone", "rock", "water", "wood"]
        }
        
        # Q-learning state-action values
        self.q_values = {}
        
        # History for plotting
        self.history = {
            "energy": [MAX_E],
            "hunger": [0],
            "pain": [0],
            "food_stored": [0],
            "actions": [],
            "rewards": []
        }
        
        self.tick_count = 0
        self.last_action = "REST"
        self.last_reward = 0
        
        self.mem0.store(self.observe())
        logging.info(f"Agent initialized at home {self.pos}.")

        self.id = agent_id if agent_id else str(id(self))
        self.learning_rate = learning_rate
        
        # Communication attributes
        self.signal_cooldown = 0  # Ticks until agent can signal again
        self.last_signal_time = 0
        self.last_signal = None
        self.signal_threshold = 0.65  # Need threshold to signal
        
        # Memory of signal observations
        self.observed_signals = []
        self.signal_outcomes = []
        
        # Perception attributes for terrain (new)
        self.last_temperature = self.w.ambient_temperature
        self.temperature_memory = []

    # ───────────── persistence helpers ─────────────
    def save_state(self, path: str = STATE_FILE):
        """Persist internal memory, learning & essential physiology."""
        np.savez_compressed(
            path,
            pos=self.pos,
            energy=self.energy,
            hunger=self.hunger,
            pain=self.pain,
            carrying=self.carrying,
            store=self.store,
            tick_count=self.tick_count,
            mem0_M=self.mem0.M,
            mem0_t=np.asarray(self.mem0.t, dtype=float),
            mem1_M=self.mem1.M,
            mem1_t=np.asarray(self.mem1.t, dtype=float),
            cell_experience=np.array([self.cell_experience], dtype=object),
            q_values=np.array([self.q_values], dtype=object),
            history=np.array([self.history], dtype=object),
            # New properties for TerrainCell-based world
            last_temperature=self.last_temperature,
            temperature_memory=np.array(self.temperature_memory, dtype=float)
        )

    def load_state(self, path: str = STATE_FILE):
        """Restore internal memory & physiology from *existing* file."""
        if not os.path.exists(path):
            logging.info("No previous state found – starting fresh.")
            return
        
        try:
            data = np.load(path, allow_pickle=True)
            
            # Load basic state
            self.pos = data["pos"].tolist()
            self.energy = float(data["energy"])
            self.hunger = float(data["hunger"])
            self.pain = float(data["pain"])
            self.carrying = bool(data["carrying"])
            self.store = int(data["store"])
            
            # Load tick count if available (backward compatibility)
            if "tick_count" in data:
                self.tick_count = int(data["tick_count"])
            else:
                logging.info("No tick_count in saved state, using default")
                
            # Load Hopfield memory
            self.mem0.M = data["mem0_M"]
            self.mem0.t = data["mem0_t"].tolist()
            self.mem1.M = data["mem1_M"]
            self.mem1.t = data["mem1_t"].tolist()
            
            # Load learning data if available (backward compatibility)
            if "cell_experience" in data:
                self.cell_experience = data["cell_experience"][0].item()
            else:
                logging.info("No experience data in saved state, using fresh experience")
                
            if "q_values" in data:
                self.q_values = data["q_values"][0].item()
            else:
                logging.info("No Q-values in saved state, starting fresh learning")
                
            if "history" in data:
                self.history = data["history"][0].item()
            else:
                # Initialize history with current state
                self.history = {
                    "energy": [self.energy],
                    "hunger": [self.hunger],
                    "pain": [self.pain],
                    "food_stored": [self.store],
                    "actions": [],
                    "rewards": []
                }
                logging.info("No history in saved state, creating new history")
                
            # Load terrain attributes if available
            if "last_temperature" in data:
                self.last_temperature = float(data["last_temperature"])
            else:
                self.last_temperature = self.w.ambient_temperature
                
            if "temperature_memory" in data:
                self.temperature_memory = data["temperature_memory"].tolist()
            else:
                self.temperature_memory = []
            
            logging.info("🔄 Agent state loaded ← %s", path)
            
        except Exception as e:
            logging.error(f"Error loading state: {e}")
            logging.info("Starting with fresh state due to load error")
            # Leave the agent with default initialization values

    # -------- experience and learning --------
    def update_experience(self, cell_type: str, reward: float):
        """Update experience for a cell type based on reward received."""
        # Get the material type for the terrain cell
        if cell_type not in self.cell_experience:
            # If it's a new material type not in our experience, add it
            self.cell_experience[cell_type] = {"reward": 0, "visits": 0, "last_visit": 0}
        
        exp = self.cell_experience[cell_type]
        if exp["visits"] == 0:
            exp["reward"] = reward
        else:
            # Weighted average favoring recent experiences
            exp["reward"] = (exp["reward"] * 0.9) + (reward * 0.1)
        exp["visits"] += 1
        exp["last_visit"] = self.tick_count
    
    def decay_experiences(self):
        """Apply decay to experiences to adapt to changing environments."""
        for cell_type in self.cell_experience:
            if self.cell_experience[cell_type]["visits"] > 0:
                self.cell_experience[cell_type]["reward"] *= EXPERIENCE_DECAY
    
    def get_state_key(self, pos: List[int], carrying: bool = None) -> str:
        """Generate a unique key for the current state."""
        carrying_val = self.carrying if carrying is None else carrying
        # Include local terrain information in the state key
        cell = self.w.cell(tuple(pos))
        return f"{tuple(pos)}|{carrying_val}|{cell.material}"
    
    def get_q_value(self, state_key: str, action: str) -> float:
        """Get Q-value for a state-action pair."""
        if state_key not in self.q_values:
            self.q_values[state_key] = {act: 0.0 for act in self.MOV.keys()}
        return self.q_values[state_key].get(action, 0.0)
    
    def update_q_value(self, state_key: str, action: str, reward: float, next_state_key: str):
        """Update Q-value using Q-learning algorithm."""
        if state_key not in self.q_values:
            self.q_values[state_key] = {act: 0.0 for act in self.MOV.keys()}
        
        # Get maximum Q-value for next state
        if next_state_key not in self.q_values:
            self.q_values[next_state_key] = {act: 0.0 for act in self.MOV.keys()}
        next_max_q = max(self.q_values[next_state_key].values())
        
        # Q-learning update formula
        old_q = self.q_values[state_key][action]
        new_q = old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max_q - old_q)
        self.q_values[state_key][action] = new_q

    # -------- perception --------
    def check_nearby_food(self) -> bool:
        """Check nearby cells for food. Return True if found."""
        # Check the current cell first
        current_cell = self.w.cell(tuple(self.pos))
        if is_food_cell(current_cell):
            return True
            
        # Look in surrounding cells for food
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = (self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID
            neighbor_cell = self.w.cell((nx, ny))
            if is_food_cell(neighbor_cell):
                return True
                
        return False
            
    def find_nearest_food_distance(self) -> int:
        """Find the distance to the nearest food cell."""
        # Check in increasing radius
        for radius in range(1, GRID // 2):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # only check perimeter
                        nx, ny = (self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID
                        cell = self.w.cell((nx, ny))
                        if is_food_cell(cell):
                            return radius
        
        return GRID  # Default to max distance if nothing found

    def observe(self) -> np.ndarray:
        """Create an observation vector from the current environment state."""
        # Initialize observation vector with zeros
        observation = np.zeros(OBS_DIM, dtype=np.float32)
        
        # Add agent's position and state
        pos_x_norm = self.pos[0] / GRID
        pos_y_norm = self.pos[1] / GRID
        energy_norm = self.energy / MAX_E
        hunger_norm = self.hunger / MAX_H
        pain_norm = self.pain / MAX_P
        carrying = 1.0 if self.carrying else 0.0
        
        # Add these basic agent states to observation
        observation[0] = pos_x_norm
        observation[1] = pos_y_norm
        observation[2] = energy_norm
        observation[3] = hunger_norm
        observation[4] = pain_norm
        observation[5] = carrying
        
        # Get current cell and surrounding cells
        center_cell = self.w.cell(tuple(self.pos))
        
        # Add cell material one-hot encoding
        cell_type = np.array([center_cell.material]).reshape(-1, 1)
        material_enc = ENC.transform(cell_type).flatten()
        for i, val in enumerate(material_enc):
            observation[6 + i] = val
        
        # Current index after material encoding
        idx = 6 + len(material_enc)
        
        # Add physical properties of current cell
        observation[idx] = center_cell.height_vector[0] / 5.0  # Normalize slope components
        observation[idx+1] = center_cell.height_vector[1] / 5.0
        observation[idx+2] = center_cell.normal_vector[0]
        observation[idx+3] = center_cell.normal_vector[1]
        observation[idx+4] = center_cell.hardness / 10.0      # Normalize material properties
        observation[idx+5] = center_cell.strength / 10.0
        observation[idx+6] = center_cell.density / 3.0
        observation[idx+7] = center_cell.friction / 3.0
        observation[idx+8] = center_cell.elasticity
        observation[idx+9] = center_cell.thermal_conductivity
        observation[idx+10] = center_cell.temperature / 40.0   # Normalize temperature (assume range 0-40°C)
        observation[idx+11] = center_cell.local_risk
        observation[idx+12] = 1.0 if center_cell.passable else 0.0
        
        # Update idx
        idx += 13
        
        # Enhanced food detection
        # Check for food in current and nearby cells
        food_nearby = self.check_nearby_food()
        observation[idx] = 1.0 if food_nearby else 0.0
        idx += 1
        
        # Search for food in visible range
        max_vision = min(20, GRID // 2)
        closest_food = None
        min_distance = float('inf')
        
        # Check in increasing radius
        for radius in range(1, max_vision + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # only check perimeter
                        nx, ny = (self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID
                        cell = self.w.cell((nx, ny))
                        
                        if is_food_cell(cell):
                            # Calculate Manhattan distance
                            distance = abs(dx) + abs(dy)
                            if distance < min_distance:
                                min_distance = distance
                                # Determine primary direction (N, S, E, W)
                                direction = None
                                if abs(dx) > abs(dy):
                                    direction = "S" if dx > 0 else "N"
                                else:
                                    direction = "E" if dy > 0 else "W"
                                closest_food = (direction, distance)
        
        # Record food distance
        if closest_food:
            # Direction encoding (one-hot)
            dir_idx = {"N": 0, "S": 1, "E": 2, "W": 3}.get(closest_food[0], 4)
            for i in range(5):  # 4 directions + none
                observation[idx + i] = 1.0 if i == dir_idx else 0.0
            
            # Distance (normalized)
            observation[idx + 5] = 1.0 - min(1.0, closest_food[1] / max_vision)
        else:
            # No food visible - leave direction as zeros
            observation[idx + 5] = 0.0  # Maximum distance (normalized to 0)
        
        idx += 6
        
        # Add environmental factors
        observation[idx] = self.w.ambient_temperature / 40.0  # Normalize temp
        observation[idx+1] = 1.0 if self.w.is_day else 0.0    # Day/night indicator
        observation[idx+2] = {"clear": 0.0, "rain": 0.5, "storm": 1.0}[self.w.weather]  # Weather encoding
        idx += 3
        
        # Add time component
        observation[idx] = (self.w.time % self.w.day_length) / self.w.day_length
        idx += 1
        
        # Distance to home
        home_x, home_y = self.w.home
        home_dist_x = abs(self.pos[0] - home_x) / GRID
        home_dist_y = abs(self.pos[1] - home_y) / GRID
        observation[idx] = home_dist_x
        observation[idx+1] = home_dist_y
        idx += 2
        
        # Reserve space at the end of the vector for food memory
        food_memory_idx = OBS_DIM - 10
        
        # If we've previously detected food and stored it in memory, enhance that signal
        if closest_food:
            # Store strong food signal in memory section
            observation[food_memory_idx] = 1.0  # Food presence marker
            observation[food_memory_idx + 1 + dir_idx] = 1.0  # Direction marker
            observation[food_memory_idx + 6] = 1.0 - min(1.0, closest_food[1] / max_vision)  # Distance marker
        
        # Pad to ensure consistent size
        while idx < food_memory_idx:
            observation[idx] = 0.0
            idx += 1
        
        # Make sure we don't exceed the observation dimension
        if idx > OBS_DIM:
            observation = observation[:OBS_DIM]
        
        return observation

    # -------- planning --------
    def plan(self) -> str:
        """Choose action based on learned experiences and exploration with path validation."""
        hunger = self.hunger / MAX_H
        pain = self.pain / MAX_P
        energy_def = (MAX_E - self.energy) / MAX_E
        current_state_key = self.get_state_key(self.pos)
        current_cell = self.w.cell(tuple(self.pos))
        
        # First, identify passable and impassable directions
        passable_directions = {}
        for act, (dx, dy) in self.MOV.items():
            if act == "REST":
                passable_directions[act] = True
                continue
            nx, ny = (self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID
            next_cell = self.w.cell((nx, ny))
            passable_directions[act] = next_cell.passable
        
        # Check if current position is passable
        if not current_cell.passable:
            # We're somehow in an impassable location, try to move to a passable adjacent cell
            for act in passable_directions:
                if act != "REST" and passable_directions[act]:
                    return act
            # If no passable cells found, just rest and hope something changes
            return "REST"
        
        # Evaluate our current plan - do we need to abandon it?
        # 1. If we're in pain, consider abandoning current plan
        if self.pain > MAX_P * 0.3:  # Even moderate pain should make us reconsider
            pain_factor = self.pain / MAX_P
            # More likely to abandon plan as pain increases
            if random.random() < pain_factor * 0.7:
                # Temporarily increase exploration to find new paths
                logging.info(f"Agent {self.id} abandoning current plan due to pain: {self.pain:.1f}")
                
                # Look for any safe direction different from last action
                safe_directions = [act for act in passable_directions 
                                if passable_directions[act] and act != self.last_action]
                
                # If we have safe options, choose the one that gets us furthest from current pain source
                if safe_directions and self.last_action != "REST":
                    # Our last action led to pain, so try to go in a different direction
                    opposite_dir = {"N": "S", "S": "N", "E": "W", "W": "E", "REST": None}
                    preferred_dir = opposite_dir.get(self.last_action)
                    
                    if preferred_dir in safe_directions:
                        logging.info(f"Agent {self.id} moving away from pain source")
                        return preferred_dir
                    else:
                        return random.choice(safe_directions)
                
                # If few safe options, consider resting to recover
                if self.rest_streak < 3:  # Don't rest too many times in a row
                    return "REST"
        
        # 2. Check if our current path is blocked ahead - "look before you leap"
        # This simulates the agent thinking a few steps ahead rather than just one
        if self.last_action not in ["REST", None] and self.last_action in passable_directions:
            # Check if continuing in the same direction would lead to an obstacle
            dx, dy = self.MOV[self.last_action]
            next_pos = [(self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID]
            next_cell = self.w.cell(tuple(next_pos))
            
            if next_cell.passable:
                # Now check one more step ahead in the same direction
                next_next_pos = [(next_pos[0] + dx) % GRID, (next_pos[1] + dy) % GRID]
                next_next_cell = self.w.cell(tuple(next_next_pos))
                
                if not next_next_cell.passable:
                    # There's an obstacle two steps ahead - consider changing direction
                    logging.info(f"Agent {self.id} detects obstacle ahead, reconsidering path")
                    
                    # Search for alternate passable directions
                    alternate_dirs = [act for act in passable_directions 
                                    if passable_directions[act] and act != self.last_action and act != "REST"]
                    
                    if alternate_dirs:
                        # If we're going toward food, home, or some other target, try to find a direction
                        # that still moves us generally in that direction
                        if self.hunger > MAX_H * 0.5 and not self.carrying:
                            # We might be heading toward food
                            for alt_dir in alternate_dirs:
                                alt_dx, alt_dy = self.MOV[alt_dir]
                                alt_next_pos = [(self.pos[0] + alt_dx) % GRID, (self.pos[1] + alt_dy) % GRID]
                                if is_food_cell(self.w.cell(tuple(alt_next_pos))):
                                    return alt_dir
                        
                        # Otherwise just pick a random alternate direction
                        return random.choice(alternate_dirs)
        
        # Enhanced food detection - prioritize going to food if hungry and not carrying food
        if self.hunger > MAX_H * 0.4 and not self.carrying:
            # Check surrounding cells for food, avoiding impassable terrain
            for act in passable_directions:
                if act == "REST" or not passable_directions[act]:
                    continue
                dx, dy = self.MOV[act]
                nx, ny = (self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID
                next_cell = self.w.cell((nx, ny))
                if is_food_cell(next_cell):
                    logging.info(f"Agent {self.id} found food, going to collect it")
                    return act
        
        # Critical energy check - force REST at home or go home when critical
        if self.energy < MAX_E * 0.15:  # Critical energy threshold
            if "home" in current_cell.tags or current_cell.material == "home":
                logging.info("CRITICAL ENERGY: Forcing rest at home")
                return "REST"
            else:
                # Head toward home when energy critical, avoiding impassable terrain
                home_x, home_y = self.w.home
                curr_x, curr_y = self.pos
                dx = home_x - curr_x
                dy = home_y - curr_y
                
                # Find best direction toward home that is passable
                directions_to_try = []
                
                # First try axes with largest distance component
                if abs(dx) > abs(dy):
                    directions_to_try.append("S" if dx > 0 else "N")
                    directions_to_try.append("E" if dy > 0 else "W")
                else:
                    directions_to_try.append("E" if dy > 0 else "W")
                    directions_to_try.append("S" if dx > 0 else "N")
                    
                # Add the remaining directions as fallbacks
                remaining = [d for d in ["N", "S", "E", "W"] if d not in directions_to_try]
                directions_to_try.extend(remaining)
                
                # Try each direction in priority order
                for direction in directions_to_try:
                    if direction in passable_directions and passable_directions[direction]:
                        return direction
                
                # If all directions are blocked, just rest
                return "REST"
        
        # Epsilon-greedy exploration (more exploration early on)
        epsilon = max(0.1, 1.0 / (1 + self.tick_count / 1000))
        
        if random.random() < epsilon:
            # Random exploration - but only consider passable directions
            valid_moves = []
            for act in passable_directions:
                if act != "REST" and passable_directions[act]:
                    valid_moves.append(act)
                    
            # If no valid moves, just rest
            if not valid_moves:
                valid_moves = ["REST"]
                
            return random.choice(valid_moves)
        
        # Calculate values for all actions
        action_values = {}
        
        for act, (dx, dy) in self.MOV.items():
            # Skip impassable directions
            if act != "REST" and not passable_directions.get(act, False):
                action_values[act] = float("-inf")  # Effectively remove this from consideration
                continue
                
            nxt = [(self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID]
            next_cell = self.w.cell(tuple(nxt))
            
            # Get Q-value component
            q_value = self.get_q_value(current_state_key, act)
            
            # Calculate combined value using Q-value and heuristics
            value = q_value
            
            # Check for food in the next cell - prioritize food gathering
            if is_food_cell(next_cell) and not self.carrying:
                value += 20.0  # Strong bonus for moving to food
            
            # Add experience-based component
            material = next_cell.material
            if material in self.cell_experience:
                exp = self.cell_experience[material]
                if exp["visits"] > 0:
                    value += exp["reward"]
                
                # Add exploration bonus for less-visited cell types
                if exp["visits"] < 10:
                    exploration_factor = 2.0 / (exp["visits"] + 1)
                    value += exploration_factor
            
            # Add terrain property considerations
            # Devalue steep terrain (higher energy cost)
            slope_steepness = np.hypot(*next_cell.height_vector)
            value -= slope_steepness * 1.5
            
            # Devalue high friction terrain
            value -= (next_cell.friction - 1.0) * 0.5
            
            # Devalue hazardous terrain
            value -= next_cell.local_risk * 10.0
            
            # Consider weather effects on terrain
            if self.w.weather == "rain" and next_cell.material in ["dirt", "stone"]:
                value -= 0.5  # Slippery when wet
            elif self.w.weather == "storm":
                value -= 1.0  # Dangerous in storms
                
            # Temperature considerations - avoid extreme temps
            temp_comfort = 1.0 - abs(next_cell.temperature - 22.0) / 20.0  # 22°C is ideal
            value += temp_comfort * 0.5
            
            # Add surprise-based exploration
            obs_nxt = self.observe() if act == "REST" else self._obs_after_move(nxt)
            surprise = self.mem0.surprise(obs_nxt)
            value += EXPLORATION_BONUS * (surprise / SURPRISE_SCALE)
            
            # Add basic heuristics
            value -= HUNGER_W * hunger + PAIN_W * pain + energy_def * 2.0
            if self.carrying and act != "REST":
                value -= CARRY_COST
            if act == "REST":
                value -= REST_PENALTY * self.rest_streak
                # Bonus for resting when energy is low
                if self.energy < MAX_E * 0.4:
                    rest_bonus = (MAX_E * 0.4 - self.energy) / MAX_E * 10.0
                    value += rest_bonus
                    # Even bigger bonus for resting at home
                    if "home" in next_cell.tags or next_cell.material == "home":
                        value += rest_bonus * 2.0
            
            # Goal-directed behavior
            if self.carrying:
                home_dist = abs(nxt[0] - self.w.home[0]) + abs(nxt[1] - self.w.home[1])
                value -= CARRY_HOME_W * (home_dist / GRID)
            if energy_def > ENERGY_LOW_FRAC:
                home_dist = abs(nxt[0] - self.w.home[0]) + abs(nxt[1] - self.w.home[1])
                value -= HOME_DIST_W * (home_dist / GRID) * (1.0 + energy_def * 2.0)
            
            action_values[act] = value
        
        # Choose the best action
        best_action = max(action_values, key=action_values.get)
        
        # Extra energy protection: if energy very low and not heading home, reconsider
        if self.energy < MAX_E * 0.25 and best_action != "REST":
            if "home" in current_cell.tags or current_cell.material == "home":
                logging.info(f"Energy low ({self.energy:.1f}): Overriding {best_action} with REST at home")
                return "REST"
        
        # Final check to ensure we don't try to move into impassable terrain
        if best_action != "REST" and not passable_directions.get(best_action, False):
            # Fallback to REST if somehow we got an impassable direction
            logging.info(f"Agent {self.id} detected impassable direction in final check, defaulting to REST")
            return "REST"
            
        return best_action

    def _obs_after_move(self, nxt: List[int]) -> np.ndarray:
        """Simulate observation after a potential move."""
        cur = self.pos
        self.pos = nxt
        obs = self.observe()
        self.pos = cur
        return obs

    # -------- acting --------
    def step(self, comm_system=None, other_agents=None):
        """Take one step in the environment and learn from it."""
        # Decrement signal cooldown if present
        if hasattr(self, 'signal_cooldown') and self.signal_cooldown > 0:
            self.signal_cooldown -= 1
        
        # Signal processing if communication system is available
        if comm_system:
            # Process incoming signals if the method exists
            if hasattr(self, 'process_signals'):
                self.process_signals(comm_system)
            
            # Check if we should send a signal
            if hasattr(self, 'should_signal') and self.should_signal(comm_system):
                if hasattr(self, 'generate_and_broadcast_signal'):
                    self.generate_and_broadcast_signal(comm_system)
            
            # Update signal outcomes from previous signals
            if hasattr(self, 'update_signal_outcomes'):
                self.update_signal_outcomes(comm_system)
            
            # Learn from signal experiences
            if hasattr(self, 'learn_from_signals'):
                self.learn_from_signals(comm_system)
            
            # Learn from observing other agents
            if other_agents and hasattr(self, 'observe_other_agents'):
                self.observe_other_agents(comm_system, other_agents)
        
        # Original step logic starts here
        self.tick_count += 1
        
        # Store previous state
        prev_pos = self.pos.copy()
        prev_carrying = self.carrying
        prev_energy = self.energy
        prev_pain = self.pain
        prev_hunger = self.hunger
        prev_state_key = self.get_state_key(prev_pos, prev_carrying)
        prev_cell = self.w.cell(tuple(prev_pos))
        
        # Choose and execute action
        act = self.plan()
        self.last_action = act
        dx, dy = self.MOV[act]
        
        # Movement or rest
        if act != "REST":
            # Calculate next position
            next_pos = [(self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID]
            next_cell = self.w.cell(tuple(next_pos))
            
            # Check if next cell is passable
            if next_cell.passable:
                self.pos = next_pos
                self.rest_streak = 0
                
                # Calculate movement cost based on terrain
                movement_cost = MOVE_COST
                
                # Consider slope/steepness
                slope_steepness = np.hypot(*next_cell.height_vector)
                movement_cost += slope_steepness * 0.5
                
                # Consider friction
                movement_cost *= next_cell.friction 
                
                # Consider density (harder materials require more energy)
                movement_cost *= (0.5 + next_cell.density * 0.5)
                
                # Apply weather effects
                if self.w.weather == "rain":
                    movement_cost *= 1.2  # Harder to move in rain
                elif self.w.weather == "storm":
                    movement_cost *= 1.5  # Much harder to move in storms
                
                # Apply carrying penalty
                if self.carrying:
                    movement_cost += CARRY_COST
                
                # Apply final movement cost
                self.energy -= movement_cost
            else:
                # Trying to move into impassable terrain
                # Cause EXTREME pain instead of movement
                pain_from_collision = PAIN_HIT * 3.0  # Triple normal pain hit
                self.pain = min(MAX_P, self.pain + pain_from_collision)
                
                # Additional severe energy penalty based on pain level
                energy_penalty = MOVE_COST * 3.0  # Base penalty
                # Add extra penalty based on current pain level - higher pain = more energy lost
                energy_penalty += (self.pain / MAX_P) * 20.0  # Up to 20 more energy lost at max pain
                
                self.energy -= energy_penalty
                
                logging.info(f"Agent {self.id} hit impassable terrain, pain: {self.pain:.1f}, energy loss: {energy_penalty:.1f}")
                
                # Force REST action since we couldn't move
                act = "REST"
                self.rest_streak += 1
        else:
            self.rest_streak += 1

        current_cell = self.w.cell(tuple(self.pos))
        
        # High pain also continuously drains energy - make pain truly impactful
        if self.pain > MAX_P * 0.7:  # If pain is above 70%
            pain_energy_drain = (self.pain / MAX_P) * 2.0  # Up to 2 energy per tick at max pain
            self.energy -= pain_energy_drain
            if pain_energy_drain > 0.5:
                logging.info(f"Agent {self.id} losing energy from high pain: {pain_energy_drain:.1f}")
        
        # Metabolism
        hunger_increase = 1.0
        # If we're resting at home, reduce the hunger increase rate
        if act == "REST" and (current_cell.material == "home" or "home" in current_cell.tags):
            hunger_increase *= 0.5  # Half the hunger increase when resting at home
        self.hunger = min(MAX_H, self.hunger + hunger_increase)
        self.pain = max(0, self.pain - PAIN_DECAY)
        
        # Base metabolism cost
        self.energy -= self.hunger / MAX_H + self.pain / MAX_P
        
        # Temperature effects on metabolism
        temp_diff = abs(current_cell.temperature - 22.0)  # Ideal temperature is 22°C
        
        # More energy spent in extreme temperatures
        if temp_diff > 10:
            self.energy -= 0.3 * (temp_diff - 10) / 10.0
        
        # Interaction with environment
        # Check for hazards first
        if current_cell.local_risk > 0:
            hazard_damage = current_cell.local_risk * PAIN_HIT
            self.pain = min(MAX_P, self.pain + hazard_damage)
        
        # Food collection and consumption
        food_collected = False
        food_eaten = False
        
        # Check if we found food
        if is_food_cell(current_cell) and not self.carrying:
            self.carrying = True
            food_collected = True
            
            # Create a strong memory trace of food location for Hopfield network
            food_obs = self.observe()
            # Strengthen the food-related components
            food_memory_idx = OBS_DIM - 10
            food_obs[food_memory_idx] = 1.0  # Strong food marker
            # Store this enhanced food observation
            self.mem0.store(food_obs)
            
            # Change the cell to dirt after collecting food
            x, y = self.pos
            food_pos = (x, y)
            
            # Create a new terrain cell of type dirt to replace the food
            dirt_cell = TerrainCell(
                height_vector=current_cell.height_vector,
                normal_vector=current_cell.normal_vector,
                material="dirt",
                passable=True,
                hardness=1.0,
                strength=1.0,
                density=1.0,
                friction=1.0,
                elasticity=0.0,
                thermal_conductivity=0.1,
                temperature=current_cell.temperature,
                local_risk=0.0,
                tags=set()
            )
            
            # Update the world grid with the new cell
            self.w.grid[x, y] = dirt_cell
            
            logging.info(f"Agent {self.id} collected food at {food_pos}")
        
        # Check for home interaction
        if (current_cell.material == "home" or "home" in current_cell.tags):
            if self.carrying and act != "REST":
                self.carrying = False
                self.store += 1
                logging.info(f"Agent {self.id} stored food at home. Total: {self.store}")
            
            if act == "REST":
                if self.carrying:
                    self.carrying = False
                    food_eaten = True
                    logging.info(f"Agent {self.id} ate carried food at home")
                elif self.store > 0:
                    self.store -= 1
                    food_eaten = True
                    logging.info(f"Agent {self.id} ate stored food at home")
                
                if food_eaten:
                    self.energy = min(MAX_E, self.energy + FOOD_E)
                    self.hunger = max(0, self.hunger - FOOD_S)
                
                # Additional rest benefits at home - only energy recovery
                self.energy = min(MAX_E, self.energy + HOME_ENERGY_RECOVERY * 0.5)
                
        # Calculate reward
        energy_change = self.energy - prev_energy
        pain_change = self.pain - prev_pain
        hunger_change = self.hunger - prev_hunger
        
        reward = energy_change - pain_change - hunger_change
        
        # Bonus for collecting or eating food
        if food_collected:
            reward += 10
        if food_eaten:
            reward += 20
        
        # Terrain-based rewards/penalties
        if current_cell.temperature > 30:
            reward -= 0.5  # Penalty for extremely hot areas
        elif current_cell.temperature < 10:
            reward -= 0.5  # Penalty for extremely cold areas
            
        # Obstacle avoidance reward
        if not prev_cell.passable and current_cell.passable:
            reward += 1.0  # Reward for getting out of impassable terrain
            
        # Day/night cycle rewards
        if not self.w.is_day and (current_cell.material == "home" or "home" in current_cell.tags):
            reward += 0.5  # Small bonus for being home at night
            
        # Weather-based rewards
        if self.w.weather == "storm" and (current_cell.material == "home" or "home" in current_cell.tags):
            reward += 1.0  # Bonus for seeking shelter during storms
        
        self.last_reward = reward
        
        # Update experience for previous cell
        self.update_experience(prev_cell.material, reward)
        
        # Update Q-values
        next_state_key = self.get_state_key(self.pos, self.carrying)
        self.update_q_value(prev_state_key, act, reward, next_state_key)
        
        # Apply experience decay
        if self.tick_count % 100 == 0:
            self.decay_experiences()
        
        # Learning via Hopfield networks
        obs = self.observe()
        
        # If the agent is hungry, try to recall food locations from memory
        if self.hunger > MAX_H * 0.5 and not self.carrying:
            recalled_obs = self.mem0.recall(obs)
            
            # Check if the recalled observation has food markers
            food_memory_idx = OBS_DIM - 10
            food_signal = recalled_obs[food_memory_idx]
            
            if food_signal > 0.7:  # Strong food signal in memory
                # This means the agent "remembers" food might be nearby
                logging.info(f"Agent {self.id} recalled food location from memory")
                
                # Use the recalled pattern to update the current observation
                # This lets the Hopfield network guide the agent toward remembered food
                # by slightly enhancing the food-related signals in the observation
                obs = obs * 0.9 + recalled_obs * 0.1
        
        # Store the potentially enhanced observation
        self.mem0.store(obs)
        
        # Handle sequence memory
        if len(self.trace) >= SEQ_LEN - 1:
            seq = np.concatenate(self.trace[-(SEQ_LEN - 1):] + [obs])
            self.mem1.store(seq)
        self.trace.append(obs)
        
        # Update temperature memory
        self.temperature_memory.append(current_cell.temperature)
        if len(self.temperature_memory) > 100:
            self.temperature_memory = self.temperature_memory[-100:]
        self.last_temperature = current_cell.temperature
        
        # Update history for visualization
        self.history["energy"].append(self.energy)
        self.history["hunger"].append(self.hunger)
        self.history["pain"].append(self.pain)
        self.history["food_stored"].append(self.store)
        self.history["actions"].append(act)
        self.history["rewards"].append(reward)
        
        # Persistence
        self.save_state()
        
        logging.debug("Tick %d – pos %s, energy %.1f, hunger %d, pain %d, reward %.1f",
                    self.tick_count, self.pos, self.energy, self.hunger, self.pain, reward)

    def find_nearest_food_direction(self) -> Tuple[bool, str, int]:
        """
        Find direction to the nearest food by leveraging the agent's existing perception systems.
        Returns: (found_food, best_direction, distance)
        """
        # First check immediate surroundings to save computation
        for act, (dx, dy) in self.MOV.items():
            if act == "REST":
                continue
            nx, ny = (self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID
            if is_food_cell(self.w.cell((nx, ny))):
                return True, act, 1
                
        # Search in larger radius
        max_vision = min(20, GRID // 2)  # Increased from previous small radius
        
        closest_food = None
        min_distance = float('inf')
        
        # Check in increasing radius
        for radius in range(2, max_vision + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # only check perimeter
                        nx, ny = (self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID
                        cell = self.w.cell((nx, ny))
                        
                        if is_food_cell(cell):
                            # Calculate Manhattan distance
                            distance = abs(dx) + abs(dy)
                            if distance < min_distance:
                                min_distance = distance
                                # Determine primary direction (N, S, E, W)
                                if abs(dx) > abs(dy):
                                    direction = "S" if dx > 0 else "N"
                                else:
                                    direction = "E" if dy > 0 else "W"
                                closest_food = (direction, distance)
        
        if closest_food:
            # Store observation of food in Hopfield network
            # Create a "food observation" pattern that encodes this information
            food_obs = self.observe()  # Get current observation
            
            # Add food location markers to the observation
            # (We'll just use the existing observation vector but modify it slightly
            # to associate it with this food location)
            food_obs = food_obs * 0.9  # Slightly dampen the current observation
            
            # Add a special food marker - we'll use indices near the end of the observation vector
            food_idx = OBS_DIM - 10
            food_dir_idx = {"N": 1, "S": 2, "E": 3, "W": 4}.get(closest_food[0], 0)
            food_obs[food_idx + food_dir_idx] = 1.0  # Mark food direction
            food_obs[food_idx + 5] = 1.0 - (closest_food[1] / max_vision)  # Mark normalized distance
            
            # Store this observation in Hopfield memory
            self.mem0.store(food_obs)
            
            return True, closest_food[0], closest_food[1]
        
        # If no food found visually, try to recall food locations from Hopfield memory
        # This is a form of "imagination" or "memory recall"
        current_obs = self.observe()
        recalled_obs = self.mem0.recall(current_obs)
        
        # Check if the recalled observation has food markers
        food_idx = OBS_DIM - 10
        food_signal = recalled_obs[food_idx:food_idx+6].sum()
        
        if food_signal > 0.5:  # If there's a significant food signal in memory
            # Extract direction from recall
            dir_values = recalled_obs[food_idx:food_idx+5]
            max_dir_idx = np.argmax(dir_values)
            if max_dir_idx > 0:  # If a valid direction was found
                directions = ["NONE", "N", "S", "E", "W"]
                recalled_dir = directions[max_dir_idx]
                
                # Extract distance from recall
                recalled_dist = (1.0 - recalled_obs[food_idx+5]) * max_vision
                
                # Calculate confidence based on pattern similarity
                confidence = self.mem0.similarity(current_obs, recalled_obs)
                
                # Only use recalled food location if confidence is high enough
                if confidence > 0.7:
                    return True, recalled_dir, int(recalled_dist)
        
        return False, None, max_vision

class AgentPopulation:
    """Manages a population of agents that can communicate and interact"""
    
    def __init__(self, world, initial_pop=5, max_pop=20):
        self.world = world
        self.agents = {}  # Dictionary with agent_id keys
        self.max_population = max_pop
        self.next_id = 0
        
        # Initialize communication system
        self.comm_system = NeuralCommunicationSystem()
        
        # Track social metrics
        self.interactions = []
        self.groups = {}
        
        # Initialize starting population
        for _ in range(initial_pop):
            self.add_agent()
            
        # Set up food sources in the environment if needed
        self.setup_food_sources()
    
    def setup_food_sources(self, num_sources=20):
        """Add food sources to the world if they don't exist"""
        # Count existing food cells
        food_count = 0
        for x in range(self.world.grid_size):
            for y in range(self.world.grid_size):
                cell = self.world.cell((x, y))
                if is_food_cell(cell):
                    food_count += 1
        
        # Add more food if needed
        if food_count < num_sources:
            for _ in range(num_sources - food_count):
                self.add_random_food()
            logging.info(f"Added food sources. Total in world: {food_count + num_sources - food_count}")
    
    def add_random_food(self):
        """Add a food source at a random passable location"""
        # Find a random passable location that isn't home or already food
        for _ in range(100):  # Try 100 times to find a suitable spot
            x = random.randint(0, self.world.grid_size - 1)
            y = random.randint(0, self.world.grid_size - 1)
            cell = self.world.cell((x, y))
            
            # Check if location is suitable
            if (cell.passable and 
                cell.material != "home" and 
                "home" not in cell.tags and
                not is_food_cell(cell)):
                
                # Create a food cell based on the current cell
                food_cell = TerrainCell(
                    height_vector=cell.height_vector,
                    normal_vector=cell.normal_vector,
                    material="food",
                    passable=True,
                    hardness=0.5,
                    strength=0.5,
                    density=0.8,
                    friction=1.0,
                    elasticity=0.0,
                    thermal_conductivity=0.1,
                    temperature=cell.temperature,
                    local_risk=0.0,
                    tags={"food"}
                )
                
                # Update the grid
                self.world.grid[x, y] = food_cell
                return True
        
        return False  # Couldn't find a suitable location
    
    def add_agent(self, parent_traits=None):
        """Create a new agent, optionally inheriting traits from parents"""
        if len(self.agents) >= self.max_population:
            return None
            
        # Create new agent
        agent_id = f"agent_{self.next_id}"
        self.next_id += 1
        
        # Create agent with optional trait inheritance
        agent = Agent(self.world, agent_id=agent_id)
        self.agents[agent_id] = agent
        
        # Initialize agent in communication system
        self.comm_system.initialize_agent(agent_id)
        
        return agent
    
    def remove_agent(self, agent_id):
        """Remove an agent from the population (death)"""
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    def step_all(self):
        """Process one time step for all agents"""
        # Clean up old signals
        self.comm_system.clean_old_signals()
        
        # Step each agent
        for agent_id, agent in self.agents.items():
            agent.step(self.comm_system, self.agents)
        
        # Handle emergent social interactions
        self.process_interactions()
        
        # Handle reproduction/death
        self.handle_population_changes()
        
        # Replenish food occasionally
        if random.random() < 0.05:  # 5% chance each tick
            self.add_random_food()
    
    def process_interactions(self):
        """Identify and process social interactions between agents"""
        # Reset interactions list
        interactions = []
        
        # Check for agents in same location
        positions = {}
        for agent_id, agent in self.agents.items():
            pos = tuple(agent.pos)
            if pos not in positions:
                positions[pos] = []
            positions[pos].append(agent_id)
        
        # For positions with multiple agents, create interaction
        for pos, agents_here in positions.items():
            if len(agents_here) > 1:
                for i in range(len(agents_here)):
                    for j in range(i+1, len(agents_here)):
                        agent1 = self.agents[agents_here[i]]
                        agent2 = self.agents[agents_here[j]]
                        
                        # Determine interaction type based on state
                        interaction = self.resolve_agent_interaction(agent1, agent2)
                        interactions.append(interaction)
        
        # Record interactions
        self.interactions.extend(interactions)
        return interactions
    
    def resolve_agent_interaction(self, agent1, agent2):
        """Handle interaction between two agents"""
        interaction = {
            "agents": [agent1.id, agent2.id],
            "position": agent1.pos.copy(),
            "time": agent1.tick_count
        }
        
        # Check if food sharing is possible
        if agent1.carrying and agent2.hunger > MAX_H * 0.7:
            # Agent1 shares food with agent2
            agent1.carrying = False
            agent2.energy = min(MAX_E, agent2.energy + FOOD_E * 0.5)
            agent2.hunger = max(0, agent2.hunger - FOOD_S * 0.5)
            
            interaction["type"] = "food_sharing"
            interaction["donor"] = agent1.id
            interaction["recipient"] = agent2.id
            
            # Reward for generous behavior
            agent1.last_reward += 5
            agent2.last_reward += 15
            
        elif agent2.carrying and agent1.hunger > MAX_H * 0.7:
            # Agent2 shares food with agent1
            agent2.carrying = False
            agent1.energy = min(MAX_E, agent1.energy + FOOD_E * 0.5)
            agent1.hunger = max(0, agent1.hunger - FOOD_S * 0.5)
            
            interaction["type"] = "food_sharing"
            interaction["donor"] = agent2.id
            interaction["recipient"] = agent1.id
            
            # Reward for generous behavior
            agent2.last_reward += 5
            agent1.last_reward += 15
            
        else:
            # Default interaction - just crossed paths
            interaction["type"] = "encounter"
        
        return interaction
    
    def handle_population_changes(self):
        """Handle agent reproduction and death"""
        # Check for deaths
        agents_to_remove = []
        for agent_id, agent in self.agents.items():
            # Death from starvation or exhaustion
            if agent.energy <= 0 or agent.hunger >= MAX_H:
                agents_to_remove.append(agent_id)
                
            # Death from extreme temperature
            #current_cell = self.world.cell(tuple(agent.pos))
            #if current_cell.temperature > 38 and random.random() < 0.1:  # 10% chance of death in extreme heat
            #    agents_to_remove.append(agent_id)
            #if current_cell.temperature < 2 and random.random() < 0.1:  # 10% chance of death in extreme cold
            #    agents_to_remove.append(agent_id)
        
        # Remove dead agents
        for agent_id in agents_to_remove:
            self.remove_agent(agent_id)
        
        # Check for reproduction opportunities
        if len(self.agents) < self.max_population:
            # Find agents with enough resources to reproduce
            viable_parents = [agent for agent in self.agents.values() 
                             if agent.energy > MAX_E * 0.7 and 
                                agent.store > 2]
            
            # Reproduction chance
            if viable_parents and random.random() < 0.05:
                parent = random.choice(viable_parents)
                
                # Create child with inherited traits
                parent_traits = {
                    "learning_rate": parent.learning_rate
                }
                
                # Parent expends resources
                parent.energy -= MAX_E * 0.3
                parent.store -= 2
                
                # Add new agent
                self.add_agent(parent_traits)
