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
from module_planner import PlanningSystem, Goal
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
        
        # Initialize memory structures for food locations
        self.known_food_locations = []  # Verified food locations
        self.false_food_locations = []  # Places we expected food but found none
        
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
        
        # Position history for detecting when agent is stuck
        self.position_history = []
        self.last_blocked_pos = None
        
        # Initialize planning system
        self.planning_system = PlanningSystem(self)
        self.current_goal = None
    
    # ───────────── persistence helpers ─────────────
    def save_state(self, path: str = STATE_FILE):
        """Persist internal memory, learning & essential physiology."""
        # Save the current goal and important planning information
        if self.current_goal:
            save_goal_state = {
                "goal_type": self.current_goal.goal_type,
                "target": self.current_goal.target,
                "priority": self.current_goal.priority,
                "plan": self.current_goal.plan,
                "plan_index": self.current_goal.plan_index
            }
        else:
            save_goal_state = {}
            
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
            temperature_memory=np.array(self.temperature_memory, dtype=float),
            # Food memory properties
            known_food_locations=np.array([self.known_food_locations], dtype=object),
            false_food_locations=np.array([self.false_food_locations], dtype=object),
            # Planning properties
            save_goal_state=np.array([save_goal_state], dtype=object),
            position_history=np.array([self.position_history], dtype=object),
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
            
            # Load food location memory
            if "known_food_locations" in data:
                self.known_food_locations = data["known_food_locations"][0].item()
            else:
                self.known_food_locations = []
                
            if "false_food_locations" in data:
                self.false_food_locations = data["false_food_locations"][0].item()
            else:
                self.false_food_locations = []
            
            # Load planning state
            if "save_goal_state" in data:
                goal_state = data["save_goal_state"][0].item()
                if goal_state and "goal_type" in goal_state:
                    # Recreate the goal
                    goal = Goal(
                        goal_type=goal_state["goal_type"],
                        target=goal_state["target"],
                        priority=goal_state["priority"]
                    )
                    goal.plan = goal_state.get("plan", [])
                    goal.plan_index = goal_state.get("plan_index", 0)
                    
                    # Set as current goal
                    self.current_goal = goal
                    self.planning_system.current_goal = goal
                    
                    # Add to goals list if not there
                    if not any(g.goal_type == goal.goal_type and g.target == goal.target 
                            for g in self.planning_system.goals):
                        self.planning_system.goals.append(goal)
            
            # Load position history
            if "position_history" in data:
                self.position_history = data["position_history"][0].item()
            else:
                self.position_history = []
            
            logging.info("🔄 Agent state loaded ← %s", path)
            
        except Exception as e:
            logging.error(f"Error loading state: {e}")
            logging.info("Starting with fresh state due to load error")
            # Leave the agent with default initialization values

    # -------- food memory management --------
    def record_food_location(self, position):
        """
        Explicitly record a verified food location with coordinates
        rather than relying solely on Hopfield memory.
        """
        # Check if this location is already recorded
        for loc in self.known_food_locations:
            if loc['position'][0] == position[0] and loc['position'][1] == position[1]:
                # Already recorded this location, update timestamp
                loc['timestamp'] = self.tick_count
                return
        
        # Store actual coordinates
        self.known_food_locations.append({
            'position': position,
            'timestamp': self.tick_count,
            'verified': True
        })
        
        # Limit the size of the food locations list
        if len(self.known_food_locations) > 20:
            self.known_food_locations = self.known_food_locations[-20:]
        
        # Also store in Hopfield memory (existing method)
        food_obs = self.observe()
        food_memory_idx = OBS_DIM - 10
        food_obs[food_memory_idx] = 1.0  # Strong food marker
        self.mem0.store(food_obs)
        
        logging.info(f"Agent {self.id} recorded verified food at {position}")

    def record_false_food_location(self, position):
        """
        Record locations where food was expected but not found,
        to avoid repeatedly going to the same incorrect location.
        """
        # Check if this location is already recorded
        for loc in self.false_food_locations:
            if loc['position'][0] == position[0] and loc['position'][1] == position[1]:
                # Already recorded this false location, update timestamp
                loc['timestamp'] = self.tick_count
                return
        
        # Record new false location
        self.false_food_locations.append({
            'position': position,
            'timestamp': self.tick_count
        })
        
        # Limit the size of the false locations list
        if len(self.false_food_locations) > 20:
            self.false_food_locations = self.false_food_locations[-20:]
        
        logging.info(f"Agent {self.id} recorded false food location at {position}")

    def check_food_at_current_location(self):
        """
        Check the current location for food and update food memory accordingly.
        Returns True if food is found, False otherwise.
        """
        current_cell = self.w.cell(tuple(self.pos))
        current_position = self.pos.copy()
        
        # Check if food exists at current location
        if is_food_cell(current_cell):
            # Found food where expected! Record this success
            self.record_food_location(current_position)
            return True
        else:
            # No food found here, check if we expected food
            if hasattr(self, 'current_goal') and self.current_goal:
                if self.current_goal.goal_type == "find_food":
                    # We were looking for food but didn't find it
                    # Mark this as a false food location
                    self.record_false_food_location(current_position)
            
            # Clean up known food locations if this place was previously marked as food
            self.known_food_locations = [
                loc for loc in self.known_food_locations
                if not (loc['position'][0] == current_position[0] and 
                      loc['position'][1] == current_position[1])
            ]
            
            return False
    
    def clean_food_memory(self):
        """Clean up stale food location data"""
        # Clean up stale false food locations (older than 1000 ticks)
        self.false_food_locations = [
            loc for loc in self.false_food_locations
            if self.tick_count - loc['timestamp'] < 1000
        ]
        
        # Clean up stale known food locations (older than 2000 ticks)
        self.known_food_locations = [
            loc for loc in self.known_food_locations
            if self.tick_count - loc['timestamp'] < 2000
        ]

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
        
        # Check if we have any known food locations
        known_food_found = False
        if self.known_food_locations:
            # Find the nearest known food location
            nearest_location = None
            nearest_distance = float('inf')
            
            for loc in self.known_food_locations:
                # Skip if in false locations
                if any(f['position'][0] == loc['position'][0] and 
                       f['position'][1] == loc['position'][1] 
                       for f in self.false_food_locations):
                    continue
                
                # Calculate distance
                distance = abs(loc['position'][0] - self.pos[0]) + abs(loc['position'][1] - self.pos[1])
                
                # Check if this is closer than current nearest
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_location = loc
            
            # If we found a nearby known food location, use it
            if nearest_location and nearest_distance < max_vision:
                # Determine direction to the location
                dx = nearest_location['position'][0] - self.pos[0]
                dy = nearest_location['position'][1] - self.pos[1]
                
                # Use it for observation
                min_distance = nearest_distance
                
                # Determine primary direction (N, S, E, W)
                if abs(dx) > abs(dy):
                    direction = "S" if dx > 0 else "N"
                else:
                    direction = "E" if dy > 0 else "W"
                
                closest_food = (direction, nearest_distance)
                known_food_found = True
        
        # If no known food locations, check in visible range
        if not known_food_found:
            # Check in increasing radius
            for radius in range(1, max_vision + 1):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if abs(dx) == radius or abs(dy) == radius:  # only check perimeter
                            nx, ny = (self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID
                            
                            # Skip locations known to be false
                            if any(f['position'][0] == nx and f['position'][1] == ny 
                                   for f in self.false_food_locations):
                                continue
                            
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
    def plan(self):
        """Choose action based on planning system and current goals."""
        # Check if we're stuck (same position for multiple steps)
        stuck = False
        if len(self.position_history) > 3:
            pos_set = set(tuple(pos) for pos in self.position_history[-3:])
            if len(pos_set) == 1 and self.last_action != "REST":
                stuck = True
        
        # Get next action from planning system
        action = self.planning_system.update()
        self.current_goal = self.planning_system.current_goal
        
        # Pre-validate the action - check if it would lead to impassable terrain
        dx, dy = self.MOV[action]
        nx, ny = (self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID
        next_cell = self.w.cell((nx, ny))
        
        # If the next cell is impassable, immediately repair plan
        if action != "REST" and not next_cell.passable:
            # Record blocked position to avoid it in future
            self.last_blocked_pos = (nx, ny)
            
            # Force plan repair immediately instead of waiting for stuck detection
            self.planning_system.repair_plan()
            
            # Try to find an alternate passable direction
            valid_dirs = []
            for alt_act, (alt_dx, alt_dy) in self.MOV.items():
                if alt_act == "REST":
                    continue
                    
                alt_nx, alt_ny = (self.pos[0] + alt_dx) % GRID, (self.pos[1] + alt_dy) % GRID
                alt_cell = self.w.cell((alt_nx, alt_ny))
                
                if alt_cell.passable and (alt_nx, alt_ny) != self.last_blocked_pos:
                    valid_dirs.append(alt_act)
            
            if valid_dirs:
                # Choose a valid direction, prioritizing directions that align with the goal
                if self.current_goal and self.current_goal.target:
                    gx, gy = self.current_goal.target
                    best_dir = None
                    best_score = float('inf')
                    
                    for vdir in valid_dirs:
                        vdx, vdy = self.MOV[vdir]
                        vnx, vny = (self.pos[0] + vdx) % GRID, (self.pos[1] + vdy) % GRID
                        score = abs(vnx - gx) + abs(vny - gy)  # Manhattan distance
                        
                        if score < best_score:
                            best_score = score
                            best_dir = vdir
                    
                    if best_dir:
                        action = best_dir
                    else:
                        # No good direction toward goal, pick a random valid one
                        action = random.choice(valid_dirs)
                else:
                    # No specific target, pick a random valid direction
                    action = random.choice(valid_dirs)
            else:
                # No valid directions, just rest for this tick
                action = "REST"
        
        # Update position history
        self.position_history.append(self.pos.copy())
        if len(self.position_history) > 10:
            self.position_history = self.position_history[-10:]
        
        return action

    def _obs_after_move(self, nxt: List[int]) -> np.ndarray:
        """Simulate observation after a potential move."""
        cur = self.pos
        self.pos = nxt
        obs = self.observe()
        self.pos = cur
        return obs

    def find_nearest_food_direction(self) -> Tuple[bool, str, int]:
        """
        Find direction to the nearest food by leveraging both direct observation
        and food memory. Prioritizes verified food locations over recalled ones.
        Returns: (found_food, best_direction, distance)
        """
        # First check immediate surroundings (direct observation)
        for act, (dx, dy) in self.MOV.items():
            if act == "REST":
                continue
            nx, ny = (self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID
            if is_food_cell(self.w.cell((nx, ny))):
                # Record this food sighting
                self.record_food_location([nx, ny])
                return True, act, 1
        
        # Check if we have any known food locations
        if self.known_food_locations:
            # Find the nearest known food location
            nearest_location = None
            nearest_distance = float('inf')
            
            for loc in self.known_food_locations:
                # Check if the location is not in our false food locations
                if any(f['position'][0] == loc['position'][0] and 
                       f['position'][1] == loc['position'][1] 
                       for f in self.false_food_locations):
                    continue  # Skip this location, it's been verified as false
                
                # Calculate distance
                distance = abs(loc['position'][0] - self.pos[0]) + abs(loc['position'][1] - self.pos[1])
                
                # Check if this is closer than current nearest
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_location = loc
            
            # If we found a nearby known food location, navigate to it
            if nearest_location and nearest_distance < GRID//2:  # Only consider if reasonably close
                # Determine direction to the location
                dx = nearest_location['position'][0] - self.pos[0]
                dy = nearest_location['position'][1] - self.pos[1]
                
                # Determine primary direction (N, S, E, W)
                if abs(dx) > abs(dy):
                    direction = "S" if dx > 0 else "N"
                else:
                    direction = "E" if dy > 0 else "W"
                
                logging.info(f"Agent {self.id} using known food location at {nearest_location['position']}")
                return True, direction, nearest_distance
        
        # If no known food locations, perform a distance search
        max_vision = min(20, GRID // 2)
        closest_food = None
        min_distance = float('inf')
        
        # Check in increasing radius
        for radius in range(2, max_vision + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # only check perimeter
                        nx, ny = (self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID
                        
                        # Skip locations known to be false
                        if any(f['position'][0] == nx and f['position'][1] == ny 
                               for f in self.false_food_locations):
                            continue
                        
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
                                closest_food = (direction, distance, [nx, ny])
        
        if closest_food:
            # Record this food sighting
            self.record_food_location(closest_food[2])
            return True, closest_food[0], closest_food[1]
        
        # As a last resort, try to recall food from Hopfield memory
        # But only use it if we don't have too many false locations
        if len(self.false_food_locations) < 5:
            current_obs = self.observe()
            recalled_obs = self.mem0.recall(current_obs)
            
            # Check if the recalled observation has food markers
            food_idx = OBS_DIM - 10
            food_signal = recalled_obs[food_idx:food_idx+6].sum()
            
            if food_signal > 0.7:  # Higher threshold for more confidence
                # Extract direction from recall
                dir_values = recalled_obs[food_idx+1:food_idx+5]
                max_dir_idx = np.argmax(dir_values)
                directions = ["N", "S", "E", "W"]
                
                if max_dir_idx < len(directions):  # Valid direction index
                    recalled_dir = directions[max_dir_idx]
                    
                    # Extract distance from recall (more conservative)
                    recalled_dist = int((1.0 - recalled_obs[food_idx+5]) * max_vision * 0.7)
                    recalled_dist = max(3, recalled_dist)  # At least 3 steps
                    
                    # Calculate confidence based on pattern similarity
                    confidence = self.mem0.similarity(current_obs, recalled_obs) if hasattr(self.mem0, 'similarity') else 0.7
                    
                    # Only use recalled food location if confidence is high enough
                    if confidence > 0.7:
                        return True, recalled_dir, int(recalled_dist)
            
            return False, None, max_vision

    def step(self, comm_system=None, agents=None):
        """Process one time step for the agent."""
        # Increment tick counter
        self.check_critical_states()
        self.tick_count += 1
        
        # Check food at current location
        self.check_food_at_current_location()
        
        # Clean up stale food memory periodically
        if self.tick_count % 100 == 0:
            self.clean_food_memory()
        
        # Perceive environment
        obs = self.observe()
        
        # Plan next action
        action = self.plan()
        
        # Retrieve new position based on action
        dx, dy = self.MOV[action]
        nx, ny = (self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID
        nxt = [nx, ny]
        
        # Get reward for this state-action
        reward = self.compute_reward(nxt, action)
        
        # Get current state key for learning
        current_state = self.get_state_key(self.pos)
        
        # Store observation in memory
        self.mem0.store(obs)
        
        # Update trace
        if len(self.trace) >= SEQ_LEN:
            self.trace.pop(0)
        self.trace.append(obs)
        
        # Update energy, hunger, pain based on action
        # Different actions have different costs
        energy_cost = MOVE_COST
        hunger_increase = 0.5  # Base hunger increase rate
        
        if action == "REST":
            # Resting has a lower energy cost
            energy_cost = MOVE_COST * 0.5
            # Hunger increases at a slower rate when resting
            hunger_increase = 0.2
            self.rest_streak += 1
        else:
            self.rest_streak = 0
        
        # Carrying adds extra cost
        if self.carrying:
            energy_cost += CARRY_COST
        
        # Update agent state
        self.energy = max(0, self.energy - energy_cost)
        self.hunger = min(MAX_H, self.hunger + hunger_increase)
        self.pain = max(0, self.pain - PAIN_DECAY)  # Natural pain decay
        
        # Check if new position is valid before moving
        next_cell = self.w.cell(tuple(nxt))
        
        if next_cell.passable:
            # Move to new position if passable
            self.pos = nxt
        else:
            # Can't move there, stay in place
            reward -= 1.0  # Penalty for trying to move to impassable terrain
            
            # Attempt to update the plan immediately when blocked
            if hasattr(self, 'planning_system') and self.planning_system and self.current_goal:
                self.planning_system.repair_plan()
        
        # Get new state key for learning update
        next_state = self.get_state_key(self.pos)
        
        # Update Q-value
        self.update_q_value(current_state, action, reward, next_state)
        
        # Update experience for current cell type
        current_cell = self.w.cell(tuple(self.pos))
        self.update_experience(current_cell.material, reward)
        
        # Handle special cell effects
        self.handle_cell_effects(current_cell)
        
        # Process environment feedback
        if action == "REST" and current_cell.material == "home":
            # Recover energy when resting at home
            self.energy = min(MAX_E, self.energy + HOME_ENERGY_RECOVERY)
                
            # Store food if carrying
            if self.carrying:
                self.carrying = False
                self.store += 1
                reward += 5  # Bonus for storing food
        
        # Check for food acquisition
        if is_food_cell(current_cell) and not self.carrying:
            # Acquire food
            self.carrying = True
            reward += 10  # Bonus for finding food
            
            # Record this food location
            self.record_food_location(self.pos)
            
            # Mark current goal as successful if looking for food
            if hasattr(self, 'current_goal') and self.current_goal and self.current_goal.goal_type == "find_food":
                self.current_goal.failed_attempts = 0
        
        # Handle sequence memory - save every few steps
        if self.tick_count % 5 == 0 and len(self.trace) == SEQ_LEN:
            self.store_sequence()
        
        # Process signals from other agents through comm system
        if comm_system and agents:
            self.process_communication(comm_system, agents)
        
        # Update agent history records
        self.history["energy"].append(self.energy)
        self.history["hunger"].append(self.hunger)
        self.history["pain"].append(self.pain)
        self.history["food_stored"].append(self.store)
        self.history["actions"].append(action)
        self.history["rewards"].append(reward)
        
        # Maintain a reasonable history length
        max_history = 1000
        if len(self.history["energy"]) > max_history:
            for key in self.history:
                if isinstance(self.history[key], list):
                    self.history[key] = self.history[key][-max_history:]
        
        # Save the last action and reward
        self.last_action = action
        self.last_reward = reward
        
        # Return the action taken
        return action
    
    def check_critical_states(self):
        """Check if agent is in a critical state requiring immediate action"""
        # Calculate critical state signals (exponential increase near limits)
        critical_hunger = False
        critical_energy = False
        critical_pain = False
        
        # Exponential signals that grow rapidly as we approach limits
        hunger_ratio = self.hunger / MAX_H
        if hunger_ratio > 0.8:
            # Exponential hunger signal (grows very fast above 80%)
            hunger_signal = 2.0 * (np.exp(3 * (hunger_ratio - 0.8)) - 1)
            if hunger_signal > 1.5:
                critical_hunger = True
        
        energy_ratio = self.energy / MAX_E
        if energy_ratio < 0.2:
            # Exponential energy depletion signal
            energy_signal = 2.0 * (np.exp(3 * (0.2 - energy_ratio)) - 1)
            if energy_signal > 1.5:
                critical_energy = True
        
        pain_ratio = self.pain / MAX_P
        if pain_ratio > 0.7:
            # Exponential pain signal
            pain_signal = 2.0 * (np.exp(3 * (pain_ratio - 0.7)) - 1)
            if pain_signal > 1.5:
                critical_pain = True
        
        # If in critical state, potentially abandon current plan
        if critical_hunger or critical_energy or critical_pain:
            if hasattr(self, 'planning_system') and self.planning_system and self.current_goal:
                # Force planning system to reconsider goals
                # If hunger is critical, prioritize food
                if critical_hunger and not self.carrying:
                    # Force a food finding goal with very high priority
                    food_goal = Goal("find_food", priority=3.0)
                    self.planning_system.goals = [g for g in self.planning_system.goals if g.goal_type != "find_food"]
                    self.planning_system.add_goal(food_goal)
                    self.planning_system.current_goal = None  # Force goal reevaluation
                
                # If carrying food with critical hunger, prioritize eating it
                elif critical_hunger and self.carrying:
                    # Create high priority goal to store/eat the food
                    store_goal = Goal("store_food", priority=3.0)
                    self.planning_system.goals = [g for g in self.planning_system.goals if g.goal_type != "store_food"]
                    self.planning_system.add_goal(store_goal)
                    self.planning_system.current_goal = None  # Force goal reevaluation
                
                # If energy is critical, prioritize returning home
                elif critical_energy:
                    # Force a return home goal with very high priority
                    home_goal = Goal("return_home", priority=3.0)
                    self.planning_system.goals = [g for g in self.planning_system.goals if g.goal_type != "return_home"]
                    self.planning_system.add_goal(home_goal)
                    self.planning_system.current_goal = None  # Force goal reevaluation
                
                # If pain is critical, prioritize rest to recover
                elif critical_pain:
                    # Force a rest goal with very high priority
                    rest_goal = Goal("rest", priority=3.0)
                    self.planning_system.goals = [g for g in self.planning_system.goals if g.goal_type != "rest"]
                    self.planning_system.add_goal(rest_goal)
                    self.planning_system.current_goal = None  # Force goal reevaluation
                
    def compute_reward(self, next_pos, action):
        """Compute reward for taking an action."""
        reward = 0.0
        
        # Apply rest penalty to discourage excessive resting
        if action == "REST":
            reward -= REST_PENALTY * (1 + 0.1 * self.rest_streak)  # Increased penalty for consecutive rests
        
        # Get current and next cell
        curr_cell = self.w.cell(tuple(self.pos))
        next_cell = self.w.cell(tuple(next_pos))
        
        # Penalty for high hunger
        hunger_penalty = (self.hunger / MAX_H) * HUNGER_W
        reward -= hunger_penalty
        
        # Penalty for pain
        pain_penalty = (self.pain / MAX_P) * PAIN_W
        reward -= pain_penalty
        
        # Reward for carrying food toward home
        if self.carrying:
            home_x, home_y = self.w.home
            curr_dist = abs(self.pos[0] - home_x) + abs(self.pos[1] - home_y)
            next_dist = abs(next_pos[0] - home_x) + abs(next_pos[1] - home_y)
            
            # Reward for moving closer to home with food
            if next_dist < curr_dist:
                reward += CARRY_HOME_W
        
        # Penalty for dangerous terrain
        risk = next_cell.local_risk
        if risk > 0:
            reward -= risk * 5
        
        # Penalty for extreme temperatures
        temp = next_cell.temperature
        if temp > 35 or temp < 5:
            reward -= 0.5  # Penalty for extreme temperature
        
        # Return calculated reward
        return reward

    def handle_cell_effects(self, cell):
        """Apply effects from the terrain cell to the agent."""
        # Apply risk effects (pain)
        if cell.local_risk > 0:
            pain_increase = cell.local_risk * 10
            self.pain = min(MAX_P, self.pain + pain_increase)
        
        # Store temperature information
        self.last_temperature = cell.temperature
        self.temperature_memory.append(cell.temperature)
        if len(self.temperature_memory) > 100:
            self.temperature_memory = self.temperature_memory[-100:]

    def store_sequence(self):
        """Store the current observation sequence in memory."""
        if len(self.trace) != SEQ_LEN:
            return
            
        # Concatenate observations to form sequence
        seq = np.concatenate(self.trace)
        
        # Store in sequence memory
        self.mem1.store(seq)

    def process_communication(self, comm_system, agents):
        """Process communication signals from other agents."""
        # Decrease signal cooldown
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
        
        # Generate and broadcast signals occasionally
        # Only send signals in certain conditions
        if self.signal_cooldown == 0:
            signal_type = None
            signal = None
            
            # Check if we're in a situation that warrants communication
            if self.carrying and random.random() < 0.3:
                # Food signal
                signal = comm_system.generate_food_signal(self)
                signal_type = "food"
            elif self.w.weather == "storm" and random.random() < 0.4:
                # Weather warning
                signal = comm_system.generate_weather_warning(self)
                signal_type = "weather"
            elif self.w.cell(tuple(self.pos)).local_risk > 0.3 and random.random() < 0.3:
                # Danger signal
                signal = comm_system.generate_terrain_warning(self)
                signal_type = "danger"
            
            # Broadcast the signal if generated
            if signal is not None and random.random() < self.signal_threshold:
                comm_system.broadcast_signal(self.id, signal, self.pos.copy())
                self.last_signal = signal
                self.last_signal_time = self.tick_count
                self.signal_cooldown = 20  # Cooldown before next signal
        
        # Perceive and interpret signals from other agents
        signals = comm_system.perceive_signals(self)
        for signal_data in signals:
            interpreted = comm_system.interpret_signal(self, signal_data)
            
            # Store interpretation
            self.observed_signals.append(interpreted)
            
            # Limit memory size
            if len(self.observed_signals) > 20:
                self.observed_signals = self.observed_signals[-20:]
            
            # Act on interpreted signal
            self.react_to_signal(interpreted)
            
            # Learn from interactions
            if signal_data["sender"] in agents:
                comm_system.learn_from_observation(self.id, signal_data["sender"])
        
        # Occasionally train communication systems if enough signal data
        if random.random() < 0.05 and len(comm_system.signal_memory.get(self.id, [])) > 10:
            comm_system.train_encoder(self.id)
            comm_system.train_decoder(self.id)

    def react_to_signal(self, interpreted_signal):
        """React to an interpreted signal from another agent."""
        meaning = interpreted_signal["meaning"]
        confidence = interpreted_signal["confidence"]
        position = interpreted_signal["position"]
        
        # Only react to high confidence signals
        if confidence < 0.6:
            return
        
        # React based on signal meaning
        if meaning == "food" and not self.carrying and self.hunger > MAX_H * 0.4:
            # Create a goal to check this location for food
            if hasattr(self, 'planning_system') and self.planning_system:
                food_goal = Goal("find_food", target=position, priority=1.2)
                self.planning_system.add_goal(food_goal)
        
        elif meaning == "danger":
            # Try to avoid this area
            if hasattr(self, 'false_food_locations'):
                # Mark as a location to avoid
                self.false_food_locations.append({
                    'position': position,
                    'timestamp': self.tick_count
                })
        
        elif meaning == "weather" and self.energy < MAX_E * 0.5:
            # Weather warning, consider heading home
            if hasattr(self, 'planning_system') and self.planning_system:
                home_goal = Goal("return_home", priority=1.0)
                self.planning_system.add_goal(home_goal)
        
        # Record outcome to help with learning
        self.signal_outcomes.append({
            "meaning": meaning,
            "position": position,
            "time": self.tick_count,
            "reaction": meaning  # Record what we did in response
        })
        
        # Limit memory size
        if len(self.signal_outcomes) > 20:
            self.signal_outcomes = self.signal_outcomes[-20:]
            
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
