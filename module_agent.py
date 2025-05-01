from __future__ import annotations
import itertools, random, time, logging, os, sys
from typing import Tuple, Dict, Any
import numpy as np
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from module_hopfield import Hopfield
from module_world import World

# ───────────────────── configuration ─────────────────────
GRID = 25
OBS_DIM = GRID * GRID * 4
SEQ_LEN, SEQ_DIM = 5, OBS_DIM * 5
CAP_L0, CAP_L1 = 800, 1200

MAX_E, MAX_H, MAX_P = 100, 100, 100
MOVE_COST, CARRY_COST = 1, 1
FOOD_E, FOOD_S = 40, 40
PAIN_HIT, PAIN_DECAY = 25, 1

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

CELL_TYPES = np.array(["home", "food", "hazard", "empty"]).reshape(-1, 1)
ENC = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(CELL_TYPES)

# ───────────────────── Agent ─────────────────────
class Agent:
    MOV = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1), "REST": (0, 0)}

    def __init__(self, w: World):
        self.w = w
        self.pos = list(w.home)
        self.energy, self.hunger, self.pain = MAX_E, 0, 0
        self.carrying, self.store = False, 0
        self.rest_streak = 0
        self.mem0 = Hopfield(OBS_DIM, CAP_L0)
        self.mem1 = Hopfield(SEQ_DIM, CAP_L1)
        self.trace: list[np.ndarray] = []
        
        # Initialize experience memory
        self.cell_experience = {
            "home": {"reward": 0, "visits": 0, "last_visit": 0},
            "food": {"reward": 0, "visits": 0, "last_visit": 0},
            "hazard": {"reward": 0, "visits": 0, "last_visit": 0},
            "empty": {"reward": 0, "visits": 0, "last_visit": 0}
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
            history=np.array([self.history], dtype=object)
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
            
            logging.info("🔄 Agent state loaded ← %s", path)
            
        except Exception as e:
            logging.error(f"Error loading state: {e}")
            logging.info("Starting with fresh state due to load error")
            # Leave the agent with default initialization values

    # -------- experience and learning --------
    def update_experience(self, cell_type: str, reward: float):
        """Update experience for a cell type based on reward received."""
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
    
    def get_state_key(self, pos: list[int], carrying: bool = None) -> str:
        """Generate a unique key for the current state."""
        carrying_val = self.carrying if carrying is None else carrying
        return f"{tuple(pos)}|{carrying_val}"
    
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
    def observe(self) -> np.ndarray:
        flat = self.w.grid.reshape(-1)
        return ENC.transform(flat[:, None]).flatten()

    # -------- planning --------
    def plan(self) -> str:
        """Choose action based on learned experiences and exploration."""
        hunger = self.hunger / MAX_H
        pain = self.pain / MAX_P
        energy_def = (MAX_E - self.energy) / MAX_E
        current_state_key = self.get_state_key(self.pos)
        food_none = self.w.nearest_food_distance(tuple(self.pos)) == GRID
        
        # Critical energy check - force REST at home or go home when critical
        if self.energy < MAX_E * 0.15:  # Critical energy threshold
            current_cell = self.w.cell(tuple(self.pos))
            if current_cell == "home":
                logging.info("CRITICAL ENERGY: Forcing rest at home")
                return "REST"
            else:
                # Head toward home when energy critical
                home_x, home_y = self.w.home
                curr_x, curr_y = self.pos
                dx = home_x - curr_x
                dy = home_y - curr_y
                
                # Choose direction with largest component
                if abs(dx) > abs(dy):
                    return "S" if dx > 0 else "N"
                else:
                    return "E" if dy > 0 else "W"
        
        # Epsilon-greedy exploration (more exploration early on)
        epsilon = max(0.1, 1.0 / (1 + self.tick_count / 1000))
        
        if random.random() < epsilon:
            # Random exploration
            return random.choice(list(self.MOV.keys()))
        
        best_action, best_value = "REST", float("-inf")
        
        for act, (dx, dy) in self.MOV.items():
            nxt = [(self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID]
            next_cell = self.w.cell(tuple(nxt))
            
            # Get Q-value component
            q_value = self.get_q_value(current_state_key, act)
            
            # Calculate combined value using Q-value and heuristics
            value = q_value
            
            # Add experience-based component if we have experience
            exp = self.cell_experience[next_cell]
            if exp["visits"] > 0:
                value += exp["reward"]
            
            # Add exploration bonus for less-visited cell types
            if exp["visits"] < 10:
                exploration_factor = 2.0 / (exp["visits"] + 1)
                value += exploration_factor
            
            # Add surprise-based exploration
            obs_nxt = self.observe() if act == "REST" else self._obs_after_move(nxt)
            surprise = self.mem0.surprise(obs_nxt)
            if food_none:
                value += EXPLORATION_BONUS * (surprise / SURPRISE_SCALE)
            
            # Add basic heuristics
            value -= HUNGER_W * hunger + PAIN_W * pain + energy_def * 2.0  # Increased energy weight
            if self.carrying and act != "REST":
                value -= CARRY_COST
            if act == "REST":
                value -= REST_PENALTY * self.rest_streak
                # Bonus for resting when energy is low
                if self.energy < MAX_E * 0.4:
                    rest_bonus = (MAX_E * 0.4 - self.energy) / MAX_E * 10.0
                    value += rest_bonus
                    # Even bigger bonus for resting at home
                    if next_cell == "home":
                        value += rest_bonus * 2.0
            
            # Goal-directed behavior
            if self.carrying:
                home_dist = abs(nxt[0] - self.w.home[0]) + abs(nxt[1] - self.w.home[1])
                value -= CARRY_HOME_W * (home_dist / GRID)
            if energy_def > ENERGY_LOW_FRAC:
                home_dist = abs(nxt[0] - self.w.home[0]) + abs(nxt[1] - self.w.home[1])
                value -= HOME_DIST_W * (home_dist / GRID) * (1.0 + energy_def * 2.0)  # Stronger home bias when energy low
            
            if value > best_value:
                best_action, best_value = act, value
        
        # Extra energy protection: if energy very low and not heading home, reconsider
        if self.energy < MAX_E * 0.25 and best_action != "REST":
            current_cell = self.w.cell(tuple(self.pos))
            if current_cell == "home":
                logging.info(f"Energy low ({self.energy:.1f}): Overriding {best_action} with REST at home")
                return "REST"
            
        return best_action

    def _obs_after_move(self, nxt: list[int]) -> np.ndarray:
        """Simulate observation after a potential move."""
        cur = self.pos
        self.pos = nxt
        obs = self.observe()
        self.pos = cur
        return obs

    # -------- acting --------
    def step(self):
        """Take one step in the environment and learn from it."""
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
            self.pos = [(self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID]
            self.rest_streak = 0
            self.energy -= MOVE_COST
            if self.carrying:
                self.energy -= CARRY_COST
        else:
            self.rest_streak += 1

        # Metabolism
        self.hunger = min(MAX_H, self.hunger + 1)
        self.pain = max(0, self.pain - PAIN_DECAY)
        self.energy -= self.hunger / MAX_H + self.pain / MAX_P

        # Interaction with environment
        curr_cell = self.w.cell(tuple(self.pos))
        food_collected = False
        food_eaten = False
        
        if curr_cell == "hazard":
            self.pain = min(MAX_P, self.pain + PAIN_HIT)
        
        if curr_cell == "food" and not self.carrying:
            self.carrying = True
            food_collected = True
            self.w.remove_food(tuple(self.pos))
        
        if curr_cell == "home":
            if self.carrying and act != "REST":
                self.carrying = False
                self.store += 1
            
            if act == "REST":
                if self.carrying:
                    self.carrying = False
                    food_eaten = True
                elif self.store > 0:
                    self.store -= 1
                    food_eaten = True
                
                if food_eaten:
                    self.energy = min(MAX_E, self.energy + FOOD_E)
                    self.hunger = max(0, self.hunger - FOOD_S)
        
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
        
        self.last_reward = reward
        
        # Update experience for previous cell
        self.update_experience(prev_cell, reward)
        
        # Update Q-values
        next_state_key = self.get_state_key(self.pos, self.carrying)
        self.update_q_value(prev_state_key, act, reward, next_state_key)
        
        # Apply experience decay
        if self.tick_count % 100 == 0:
            self.decay_experiences()
        
        # Learning via Hopfield networks
        obs = self.observe()
        self.mem0.store(obs)
        if len(self.trace) >= SEQ_LEN - 1:
            seq = np.concatenate(self.trace[-(SEQ_LEN - 1):] + [obs])
            self.mem1.store(seq)
        self.trace.append(obs)
        
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
