from __future__ import annotations
import itertools, random, time, logging, os, sys
from typing import Tuple, Dict, Any, List, Optional, Set, Callable
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from module_hopfield import Hopfield
from module_new_world import World, TerrainCell
from module_planner import PlanningSystem

# ───────────────────── Configuration ─────────────────────
# Import constants from a separate file to avoid duplication
try:
    from constants import *
except ImportError:
    # Default constants if constants.py not available
    GRID = 40
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
    """
    Autonomous agent that interacts with a world environment.
    
    Features:
    - Memory systems based on Hopfield networks
    - Learning mechanisms for adapting to environmental changes
    - Signal-based planning system for intelligent decision making
    - Energy, hunger, and pain management
    """
    
    # Movement actions and their corresponding direction vectors
    MOV = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1), "REST": (0, 0)}

    def __init__(self, world, agent_id=None, learning_rate=LEARNING_RATE):
        """
        Initialize a new agent in the world
        
        Args:
            world: World instance the agent will interact with
            agent_id: Optional identifier for the agent
            learning_rate: Learning rate for Q-learning updates
        """
        self.w = world
        self.pos = list(world.home)
        self.energy, self.hunger, self.pain = MAX_E, 0, 0
        self.carrying, self.store = False, 0
        self.rest_streak = 0
        self.mem0 = Hopfield(OBS_DIM, CAP_L0)  # Short-term memory
        self.mem1 = Hopfield(SEQ_DIM, CAP_L1)  # Long-term memory 
        self.trace: list[np.ndarray] = []
        
        # Initialize experience memory for different terrain types
        self.cell_experience = {}
        for material in ["home", "food", "dirt", "stone", "rock", "water", "wood"]:
            self.cell_experience[material] = {
                "reward": 0, 
                "visits": 0, 
                "last_visit": 0
            }
        
        # Q-learning state-action values
        self.q_values = {}
        
        # History for plotting and analysis
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
        
        # Store initial observation in memory
        self.mem0.store(self.observe())
        
        # Set agent identifier
        self.id = agent_id if agent_id else f"agent_{id(self)}"
        self.learning_rate = learning_rate
        
        # Sensors and perception
        self.last_temperature = self.w.ambient_temperature
        self.temperature_memory = []
        
        # Position history for detecting when agent is stuck
        self.position_history = []
        self.last_blocked_pos = None
        
        # Initialize planning system
        self.planning_system = PlanningSystem(self)
        
        logging.info(f"Agent {self.id} initialized at home {self.pos}.")

    # ───────────── State Management ─────────────
    def save_state(self, path: str = STATE_FILE):
        """
        Persist agent state to a file including memory, learning data, and physiology.
        
        Args:
            path: Path where state will be saved
        """
        # Save the current plan information
        plan_state = {
            "current_plan": self.planning_system.current_plan,
            "plan_index": self.planning_system.plan_index,
            "current_target": self.planning_system.current_target,
            "consecutive_failures": self.planning_system.consecutive_failures
        }
            
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
            plan_state=np.array([plan_state], dtype=object),
            position_history=np.array([self.position_history], dtype=object),
        )
        
    def load_state(self, path: str = STATE_FILE):
        """
        Restore agent state from a saved file.
        
        Args:
            path: Path to the saved state file
        """
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
            if "plan_state" in data:
                plan_state = data["plan_state"][0].item()
                if plan_state:
                    # Restore planning system state
                    self.planning_system.current_plan = plan_state.get("current_plan", [])
                    self.planning_system.plan_index = plan_state.get("plan_index", 0)
                    self.planning_system.current_target = plan_state.get("current_target", None)
                    self.planning_system.consecutive_failures = plan_state.get("consecutive_failures", 0)
            
            # Load position history
            if "position_history" in data:
                self.position_history = data["position_history"][0].item()
            else:
                self.position_history = []
            
            logging.info(f"🔄 Agent {self.id} state loaded from {path}")
            
        except Exception as e:
            logging.error(f"Error loading state: {e}")
            logging.info("Starting with fresh state due to load error")
            # Leave the agent with default initialization values

    # ───────────── Memory and Learning ─────────────
    def record_food_location(self, position):
        """
        Record a verified food location to improve future navigation.
        
        Args:
            position: [x, y] coordinates of the food
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
        Record locations where food was expected but not found.
        
        Args:
            position: [x, y] coordinates that don't contain food
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
        Check the current location for food and update food memory.
        
        Returns:
            bool: True if food is found, False otherwise
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
            if self.planning_system.current_target:
                current_x, current_y = current_position
                target_x, target_y = self.planning_system.current_target
                
                # If we're at our target location but no food, record as false
                if current_x == target_x and current_y == target_y:
                    self.record_false_food_location(current_position)
            
            # Clean up known food locations if this place was previously marked as food
            self.known_food_locations = [
                loc for loc in self.known_food_locations
                if not (loc['position'][0] == current_position[0] and 
                      loc['position'][1] == current_position[1])
            ]
            
            return False
    
    def clean_food_memory(self):
        """Clean up stale food location data from memory"""
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

    def update_experience(self, cell_type: str, reward: float):
        """
        Update experience for a cell type based on reward received.
        
        Args:
            cell_type: Type of terrain cell
            reward: Reward value for this experience
        """
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
        """Apply decay to experiences to adapt to changing environments"""
        for cell_type in self.cell_experience:
            if self.cell_experience[cell_type]["visits"] > 0:
                self.cell_experience[cell_type]["reward"] *= EXPERIENCE_DECAY
    
    def get_state_key(self, pos: List[int], carrying: bool = None) -> str:
        """
        Generate a unique key for the current state.
        
        Args:
            pos: Position coordinates [x, y]
            carrying: Whether agent is carrying food (default: use agent's current state)
            
        Returns:
            str: Unique state identifier for Q-learning
        """
        carrying_val = self.carrying if carrying is None else carrying
        # Include local terrain information in the state key
        cell = self.w.cell(tuple(pos))
        return f"{tuple(pos)}|{carrying_val}|{cell.material}"
    
    def get_q_value(self, state_key: str, action: str) -> float:
        """
        Get Q-value for a state-action pair.
        
        Args:
            state_key: State identifier 
            action: Action string ("N", "S", "E", "W", "REST")
            
        Returns:
            float: Q-value for the state-action pair
        """
        if state_key not in self.q_values:
            self.q_values[state_key] = {act: 0.0 for act in self.MOV.keys()}
        return self.q_values[state_key].get(action, 0.0)
    
    def update_q_value(self, state_key: str, action: str, reward: float, next_state_key: str):
        """
        Update Q-value using Q-learning algorithm.
        
        Args:
            state_key: Current state identifier
            action: Action taken
            reward: Reward received
            next_state_key: Next state identifier
        """
        if state_key not in self.q_values:
            self.q_values[state_key] = {act: 0.0 for act in self.MOV.keys()}
        
        # Get maximum Q-value for next state
        if next_state_key not in self.q_values:
            self.q_values[next_state_key] = {act: 0.0 for act in self.MOV.keys()}
        next_max_q = max(self.q_values[next_state_key].values())
        
        # Q-learning update formula
        old_q = self.q_values[state_key][action]
        new_q = old_q + self.learning_rate * (reward + DISCOUNT_FACTOR * next_max_q - old_q)
        self.q_values[state_key][action] = new_q

    # ───────────── Perception and Sensing ─────────────
    def check_nearby_food(self) -> bool:
        """
        Check nearby cells for food.
        
        Returns:
            bool: True if food is found in the current or adjacent cells
        """
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
        """
        Find the distance to the nearest food cell.
        
        Returns:
            int: Manhattan distance to nearest food or GRID if none found
        """
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
        """
        Create an observation vector from the current environment state.
        
        Returns:
            np.ndarray: Vector representation of the agent's observation
        """
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
        
    def find_nearest_food_direction(self) -> Tuple[bool, Optional[str], int, Optional[List[int]]]:
        """
        Find direction to the nearest food by leveraging both direct observation
        and food memory.
        
        Returns:
            Tuple[bool, Optional[str], int, Optional[List[int]]]: (found_food, best_direction, distance, coordinates)
        """
        # First check immediate surroundings (direct observation)
        for act, (dx, dy) in self.MOV.items():
            if act == "REST":
                continue
            nx, ny = (self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID
            if is_food_cell(self.w.cell((nx, ny))):
                # Record this food sighting
                self.record_food_location([nx, ny])
                print(f"Agent {self.id}: Found adjacent food at coordinates [{nx}, {ny}]")
                return True, act, 1, [nx, ny]
        
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
                # CRITICAL FIX: Directions should be properly matched to changes on each axis
                # N/S: Change in x-axis, E/W: Change in y-axis
                if abs(dx) > abs(dy):
                    direction = "S" if dx > 0 else "N"
                else:
                    direction = "E" if dy > 0 else "W"
                
                print(f"Agent {self.id}: Using known food location at {nearest_location['position']}, distance {nearest_distance}, direction {direction}")
                return True, direction, nearest_distance, nearest_location['position']
        
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
            print(f"Agent {self.id}: Found distant food at {closest_food[2]}, direction {closest_food[0]}, distance {closest_food[1]}")
            return True, closest_food[0], closest_food[1], closest_food[2]
        
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
                        # We don't have exact coordinates for recalled food
                        print(f"Agent {self.id}: Using recalled food memory, direction {recalled_dir}, distance {recalled_dist}")
                        return True, recalled_dir, int(recalled_dist), None
        
        return False, None, max_vision, None

    # ───────────── Planning and Decision Making ─────────────
    def plan(self):
        """
        Choose action based on signal-driven planning system.
        
        Returns:
            str: The selected action
        """
        # Check if we're stuck (same position for multiple steps)
        stuck = False
        if len(self.position_history) > 3:
            pos_set = set(tuple(pos) for pos in self.position_history[-3:])
            if len(pos_set) == 1 and self.last_action != "REST":
                stuck = True
                # If we're stuck, increment the consecutive failures counter
                self.planning_system.consecutive_failures += 1
        
        # Get next action from planning system
        action = self.planning_system.update()
        
        # Pre-validate the action - check if it would lead to impassable terrain
        dx, dy = self.MOV[action]
        nx, ny = (self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID
        next_cell = self.w.cell((nx, ny))
        
        # If the next cell is impassable and it's a movement action, find alternate
        if action != "REST" and not next_cell.passable:
            # Record blocked position to avoid it in future
            self.last_blocked_pos = (nx, ny)
            
            # Increment the consecutive failures counter, but don't immediately repair
            # - we want some persistence before abandoning the plan
            self.planning_system.consecutive_failures += 1
            
            # Only repair the plan if we've hit the failure threshold
            if self.planning_system.consecutive_failures >= 3:
                self.planning_system.repair_plan()
                # Get new action after repair
                action = self.planning_system.update()
            else:
                # Find an alternative direction just for this step
                valid_dirs = []
                for alt_act, (alt_dx, alt_dy) in self.MOV.items():
                    if alt_act == "REST":
                        continue
                        
                    alt_nx, alt_ny = (self.pos[0] + alt_dx) % GRID, (self.pos[1] + alt_dy) % GRID
                    alt_cell = self.w.cell((alt_nx, alt_ny))
                    
                    if alt_cell.passable and (alt_nx, alt_ny) != self.last_blocked_pos:
                        valid_dirs.append(alt_act)
                
                if valid_dirs:
                    # Choose a valid direction, prioritizing ones closest to original direction
                    if self.planning_system.current_target:
                        target_x, target_y = self.planning_system.current_target
                        best_dir = None
                        best_score = float('inf')
                        
                        for vdir in valid_dirs:
                            vdx, vdy = self.MOV[vdir]
                            vnx, vny = (self.pos[0] + vdx) % GRID, (self.pos[1] + vdy) % GRID
                            score = abs(vnx - target_x) + abs(vny - target_y)  # Manhattan distance
                            
                            if score < best_score:
                                best_score = score
                                best_dir = vdir
                        
                        if best_dir:
                            action = best_dir
                        else:
                            # No good direction toward target, pick a random valid one
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
        """
        Simulate observation after a potential move.
        
        Args:
            nxt: Next position [x, y]
            
        Returns:
            np.ndarray: Simulated observation vector
        """
        cur = self.pos
        self.pos = nxt
        obs = self.observe()
        self.pos = cur
        return obs
        
    def assess_internal_signals(self):
        """
        Assess the agent's internal signals (hunger, energy, pain) to determine action needs
        
        Returns:
            dict: Internal signals with their normalized values and urgency
        """
        # Calculate normalized signals
        hunger_ratio = self.hunger / MAX_H  
        energy_ratio = self.energy / MAX_E
        pain_ratio = self.pain / MAX_P
        
        # Calculate signal strengths with mild emphasis on extreme values
        hunger_signal = hunger_ratio * (1.0 + hunger_ratio)  # Non-linear scaling
        energy_signal = (1.0 - energy_ratio) * (1.0 + (1.0 - energy_ratio))  # Invert and scale
        pain_signal = pain_ratio * (1.0 + pain_ratio)
        
        # Create signal dictionary
        signals = {
            "hunger": {
                "value": hunger_ratio,
                "urgency": hunger_signal
            },
            "energy": {
                "value": energy_ratio,
                "urgency": energy_signal
            },
            "pain": {
                "value": pain_ratio,
                "urgency": pain_signal
            }
        }
        
        # Add carrying status
        signals["carrying"] = {
            "value": 1.0 if self.carrying else 0.0,
            "urgency": 0.5 if self.carrying else 0.0
        }
        
        return signals
        
    def assess_external_signals(self):
        """
        Assess external environment signals
        
        Returns:
            dict: External signals with their values
        """
        # Get current cell
        current_cell = self.w.cell(tuple(self.pos))
        
        # Calculate distance to home
        home_x, home_y = self.w.home
        home_dist = abs(self.pos[0] - home_x) + abs(self.pos[1] - home_y)
        home_dist_ratio = min(1.0, home_dist / (GRID / 2))
        
        # Check for food
        food_nearby = self.check_nearby_food()
        food_dist = self.find_nearest_food_distance()
        food_dist_ratio = min(1.0, food_dist / (GRID / 2))
        
        # Create external signals dictionary
        signals = {
            "home_distance": {
                "value": home_dist_ratio,
                "urgency": home_dist_ratio * 0.5  # Less urgent than internal signals
            },
            "food_nearby": {
                "value": 1.0 if food_nearby else 0.0,
                "urgency": 0.8 if food_nearby else 0.0
            },
            "food_distance": {
                "value": food_dist_ratio,
                "urgency": (1.0 - food_dist_ratio) * 0.5  # Closer food = higher urgency
            },
            "terrain_risk": {
                "value": current_cell.local_risk,
                "urgency": current_cell.local_risk * 0.7
            },
            "at_home": {
                "value": 1.0 if current_cell.material == "home" or "home" in current_cell.tags else 0.0,
                "urgency": 0.0  # Being at home itself isn't urgent
            },
            "weather": {
                "value": {"clear": 0.0, "rain": 0.5, "storm": 1.0}[self.w.weather],
                "urgency": {"clear": 0.0, "rain": 0.2, "storm": 0.6}[self.w.weather]
            }
        }
        
        return signals

    # ───────────── Reward System ─────────────
    def compute_reward(self, next_pos, action):
        """
        Compute reward for taking an action.
        
        Args:
            next_pos: The position after taking action
            action: The action taken
            
        Returns:
            float: Reward value
        """
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
        """
        Apply effects from the terrain cell to the agent.
        
        Args:
            cell: Current TerrainCell
        """
        # Apply risk effects (pain)
        if cell.local_risk > 0:
            pain_increase = cell.local_risk * 10
            self.pain = min(MAX_P, self.pain + pain_increase)
        
        # Store temperature information
        self.last_temperature = cell.temperature
        self.temperature_memory.append(cell.temperature)
        if len(self.temperature_memory) > 100:
            self.temperature_memory = self.temperature_memory[-100:]

    # ───────────── Memory Storage ─────────────
    def store_sequence(self):
        """Store the current observation sequence in memory."""
        if len(self.trace) != SEQ_LEN:
            return
            
        # Concatenate observations to form sequence
        seq = np.concatenate(self.trace)
        
        # Store in sequence memory
        self.mem1.store(seq)

    # ───────────── Simple Communication System ─────────────
    def query_nearby_agents(self, nearby_agents):
        """
        Query nearby agents for information based on current needs
        
        Args:
            nearby_agents: List of nearby agent objects
        """
        # Determine most pressing need
        internal_signals = self.assess_internal_signals()
        
        # No need to query if no needs or no other agents
        if not nearby_agents or len(nearby_agents) == 0:
            return
            
        # Find the most urgent need
        most_urgent = max(internal_signals.items(), key=lambda x: x[1]["urgency"])
        need_type, need_info = most_urgent
        
        # Only query if need is significant
        if need_info["urgency"] < 0.5:
            return
            
        query_results = []
        
        # Ask each nearby agent
        for agent in nearby_agents:
            # Skip self
            if agent.id == self.id:
                continue
                
            if need_type == "hunger":
                # Ask about food
                food_info = self.query_for_food(agent)
                if food_info:
                    query_results.append(food_info)
                    
            elif need_type == "energy":
                # Ask about safe paths home
                home_info = self.query_for_home(agent)
                if home_info:
                    query_results.append(home_info)
                        
        # Process and use the information
        if query_results:
            self.use_query_results(need_type, query_results)
            
    def query_for_food(self, other_agent):
        """
        Query another agent for food information
        
        Args:
            other_agent: Agent to query
            
        Returns:
            dict: Food information if available, None otherwise
        """
        # Check if other agent has known food locations
        if hasattr(other_agent, 'known_food_locations') and other_agent.known_food_locations:
            # Find nearest known location
            nearest_food = None
            nearest_dist = float('inf')
            
            for loc in other_agent.known_food_locations:
                dist = abs(loc['position'][0] - self.pos[0]) + abs(loc['position'][1] - self.pos[1])
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_food = loc
                    
            if nearest_food:
                return {
                    "type": "food_location",
                    "position": nearest_food['position'],
                    "distance": nearest_dist,
                    "timestamp": nearest_food['timestamp']
                }
                
        return None
            
    def query_for_home(self, other_agent):
        """
        Query another agent for safe path home information
        
        Args:
            other_agent: Agent to query
            
        Returns:
            dict: Path information if available, None otherwise
        """
        # Check if other agent has traveled home recently
        if hasattr(other_agent, 'position_history') and other_agent.position_history:
            # The other agent's position history could contain info about safe paths
            # A simple approach: check if their history contains a direct path to home
            
            # Check if the history contains home
            home_x, home_y = self.w.home
            home_in_history = False
            
            for pos in other_agent.position_history:
                if pos[0] == home_x and pos[1] == home_y:
                    home_in_history = True
                    break
                    
            if home_in_history:
                # Extract a path segment
                path_segment = []
                for i in range(min(5, len(other_agent.position_history))):
                    path_segment.append(other_agent.position_history[i])
                    
                return {
                    "type": "home_path",
                    "path_segment": path_segment
                }
                
        return None
            
    def use_query_results(self, need_type, results):
        """
        Use the information obtained from queries
        
        Args:
            need_type: Type of need that prompted the query
            results: List of results from queries
        """
        if not results:
            return
            
        if need_type == "hunger":
            # Use food location information
            food_locations = [r for r in results if r["type"] == "food_location"]
            
            if food_locations:
                # Sort by distance
                food_locations.sort(key=lambda x: x["distance"])
                
                # Record closest food location
                closest = food_locations[0]
                self.record_food_location(closest["position"])
                
                # Update planning system if we have one
                if hasattr(self, 'planning_system') and self.planning_system:
                    self.planning_system.set_target(closest["position"])
                    
        elif need_type == "energy":
            # Use home path information
            home_paths = [r for r in results if r["type"] == "home_path"]
            
            if home_paths:
                # Take the first path info (could be improved by evaluating paths)
                path_info = home_paths[0]
                
                # Update planning system if we have one
                if hasattr(self, 'planning_system') and self.planning_system:
                    self.planning_system.consider_path(path_info["path_segment"])

    # ───────────── Main Step Function ─────────────  
    def step(self, nearby_agents=None):
        """
        Process one time step for the agent.
        
        Args:
            nearby_agents: Optional list of nearby agents for communication
            
        Returns:
            str: The action taken
        """
        # Increment tick counter
        self.tick_count += 1
        
        # Check food at current location
        food_at_location = self.check_food_at_current_location()
        
        # Clean up stale food memory periodically
        if self.tick_count % 100 == 0:
            self.clean_food_memory()
        
        # Gather internal and external signals
        internal_signals = self.assess_internal_signals()
        external_signals = self.assess_external_signals()
        
        # Query nearby agents if available and needed
        if nearby_agents and len(nearby_agents) > 0:
            # Only query if we have a significant need
            if (internal_signals["hunger"]["urgency"] > 0.7 or 
                internal_signals["energy"]["urgency"] > 0.7):
                self.query_nearby_agents(nearby_agents)
                    
        # Perceive environment
        obs = self.observe()
        
        # Plan next action based on signals
        action = self.plan()
        
        # CRITICAL: Check if we're at a food cell and we're not carrying food
        # If so, pick up the food if the action is REST
        current_cell = self.w.cell(tuple(self.pos))
        if is_food_cell(current_cell) and not self.carrying and action == "REST":
            print(f"Agent {self.id}: Picking up food at {self.pos}!")
            self.carrying = True
            # Record successful food pickup
            self.record_food_location(self.pos)
            
            # ADDED: Modify the world grid to remove the food
            # Create a regular dirt cell to replace the food
            new_cell = TerrainCell(
                height_vector=current_cell.height_vector,
                normal_vector=current_cell.normal_vector,
                material="dirt",  # Change material from food to dirt
                passable=True,
                hardness=current_cell.hardness,
                strength=current_cell.strength,
                density=current_cell.density,
                friction=current_cell.friction,
                elasticity=current_cell.elasticity,
                thermal_conductivity=current_cell.thermal_conductivity,
                temperature=current_cell.temperature,
                local_risk=current_cell.local_risk,
                tags=set()  # Remove any food tags
            )
            # Update the world grid at this position
            self.w.grid[self.pos[0], self.pos[1]] = new_cell
            
            # Update planning system
            if hasattr(self, 'planning_system'):
                self.planning_system.plan_success()
        
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
        
        # Update hunger, pain and possibly energy based on action
        hunger_increase = 0.5  # Base hunger increase rate

        if action == "REST":
            # When resting - no energy cost and slower hunger increase
            energy_cost = 0  # No energy cost when resting
            
            # Slow energy recovery when resting
            energy_recovery = 0.2  # Small amount of energy recovery when resting
            
            # Hunger increases at a slower rate when resting
            hunger_increase = 0.2
            self.rest_streak += 1
            
            # Apply energy recovery
            self.energy = min(MAX_E, self.energy + energy_recovery)
        else:
            # Moving consumes energy
            energy_cost = MOVE_COST
            
            # Carrying adds extra cost, but only when moving
            if self.carrying:
                energy_cost += CARRY_COST
                
            # Apply energy cost for movement
            self.energy = max(0, self.energy - energy_cost)
            
            # Reset rest streak
            self.rest_streak = 0

        # Always update hunger and pain
        self.hunger = min(MAX_H, self.hunger + hunger_increase)
        self.pain = max(0, self.pain - PAIN_DECAY)  # Natural pain decay
        
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
            
            # Increment failures counter on the planning system but don't immediately repair
            # - we want some persistence before abandoning the plan
            if hasattr(self, 'planning_system'):
                self.planning_system.consecutive_failures += 1
                
                # Only repair if we've failed multiple times
                if self.planning_system.consecutive_failures >= 3:
                    self.planning_system.repair_plan()
        
        # Get new state key for learning
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
                print(f"Agent {self.id}: Successfully stored food at home!")
                
                # Mark storage goal as successful
                if hasattr(self, 'planning_system'):
                    self.planning_system.plan_success()
        
        # Handle sequence memory - save every few steps
        if self.tick_count % 5 == 0 and len(self.trace) == SEQ_LEN:
            self.store_sequence()
        
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

# ───────────────────── Agent Population ─────────────────────
class AgentPopulation:
    """Manages a population of agents that can communicate and interact"""
    
    def __init__(self, world, initial_pop=5, max_pop=20):
        """
        Initialize a population of agents in the world.
        
        Args:
            world: World instance
            initial_pop: Initial number of agents
            max_pop: Maximum number of agents allowed
        """
        self.world = world
        self.agents = {}  # Dictionary with agent_id keys
        self.max_population = max_pop
        self.next_id = 0
        
        # Track social metrics
        self.interactions = []  # List to track agent interactions
        self.groups = {}  # Dictionary to track social groups
        
        # Initialize a dummy communication system to handle the case where PyTorch is not available
        # This will be replaced with a real comm system if PyTorch is available
        class DummyCommSystem:
            def __init__(self):
                self.active_signals = []
            def clean_old_signals(self):
                pass
            def initialize_agent(self, agent_id):
                pass
        
        # Initialize the communication system
        self.comm_system = DummyCommSystem()
        
        # Initialize starting population
        for _ in range(initial_pop):
            self.add_agent()
            
        # Set up food sources in the environment if needed
        self.setup_food_sources()
    
    def setup_food_sources(self, num_sources=20):
        """
        Add food sources to the world if they don't exist.
        
        Args:
            num_sources: Target number of food sources
        """
        # Helper function to check if a cell contains food
        def is_food_cell(cell):
            return (cell.material == "food" or "food" in cell.tags)
            
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
        # Helper function to check if a cell contains food
        def is_food_cell(cell):
            return (cell.material == "food" or "food" in cell.tags)
            
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
        """
        Create a new agent, optionally inheriting traits from parents.
        
        Args:
            parent_traits: Optional traits to inherit
            
        Returns:
            Agent: Newly created agent or None if max population reached
        """
        if len(self.agents) >= self.max_population:
            return None
            
        # Create new agent
        agent_id = f"agent_{self.next_id}"
        self.next_id += 1
        
        # Create agent with optional trait inheritance
        agent = Agent(self.world, agent_id=agent_id)
        self.agents[agent_id] = agent
        
        # Initialize agent in communication system if available
        if hasattr(self.comm_system, 'initialize_agent'):
            self.comm_system.initialize_agent(agent_id)
        
        return agent
    
    def remove_agent(self, agent_id):
        """
        Remove an agent from the population (death)
        
        Args:
            agent_id: ID of agent to remove
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    def step_all(self):
        """Process one time step for all agents"""
        # Helper function to check if a cell contains food
        def is_food_cell(cell):
            return (cell.material == "food" or "food" in cell.tags)
            
        # Gather all agents at each position for communication
        positions = {}
        for agent_id, agent in self.agents.items():
            pos = tuple(agent.pos)
            if pos not in positions:
                positions[pos] = []
            positions[pos].append(agent)
            
        # Step each agent with nearby agents for communication
        for agent_id, agent in self.agents.items():
            # Find nearby agents (in same or adjacent cells)
            nearby = []
            agent_pos = tuple(agent.pos)
            
            # Add agents in the same cell
            if agent_pos in positions:
                nearby.extend([a for a in positions[agent_pos] if a.id != agent_id])
            
            # Add agents in adjacent cells
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                adj_pos = ((agent_pos[0] + dx) % self.world.grid_size, 
                          (agent_pos[1] + dy) % self.world.grid_size)
                if adj_pos in positions:
                    nearby.extend(positions[adj_pos])
            
            # Step the agent with nearby agents for communication
            agent.step(nearby)
            
            # Record basic interaction when agents are in the same cell
            for other in nearby:
                if other.id == agent_id:
                    continue
                
                # Record a basic proximity interaction
                if random.random() < 0.1:  # Only record some interactions to avoid spam
                    self.record_interaction(agent_id, other.id, "proximity", agent_pos)
            
        # Update food sources periodically
        if random.random() < 0.01:  # 1% chance per step
            self.update_food_sources()
        
        # Clean up stale communication signals
        self.comm_system.clean_old_signals()
    
    def record_interaction(self, agent1_id, agent2_id, interaction_type, position):
        """
        Record an interaction between two agents
        
        Args:
            agent1_id: ID of first agent
            agent2_id: ID of second agent
            interaction_type: Type of interaction
            position: Position where interaction occurred
        """
        self.interactions.append({
            "type": interaction_type,
            "agents": [agent1_id, agent2_id],
            "tick": self.world.time,
            "position": position
        })
        
        # Limit the number of stored interactions
        if len(self.interactions) > 1000:
            self.interactions = self.interactions[-1000:]
    
    def update_food_sources(self, replenish_chance=0.01, max_food=20):
        """
        Update food sources in the world, with chance to replenish consumed ones
        
        Args:
            replenish_chance: Probability of adding new food in each step
            max_food: Maximum number of food sources in the world
        """
        # Helper function to check if a cell contains food
        def is_food_cell(cell):
            return (cell.material == "food" or "food" in cell.tags)
            
        # Count current food sources
        food_count = 0
        for x in range(self.world.grid_size):
            for y in range(self.world.grid_size):
                cell = self.world.cell((x, y))
                if is_food_cell(cell):
                    food_count += 1
        
        # Check if we need to add more food
        if food_count < max_food and random.random() < replenish_chance:
            self.add_random_food()
