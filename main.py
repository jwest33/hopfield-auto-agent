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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S",
                    stream=sys.stdout)

# ───────────────────── encoder ─────────────────────
CELL_TYPES = np.array(["home", "food", "hazard", "empty"]).reshape(-1, 1)
ENC = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(CELL_TYPES)

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

# ───────────────────── World ─────────────────────
class World:
    def __init__(self):
        logging.info("Creating new world grid …")
        self.grid = np.full((GRID, GRID), "empty", object)
        self.home = (GRID // 2, GRID // 2)
        self.grid[self.home] = "home"
        for _ in range(80):
            self._rand("food")
        for _ in range(70):
            self._rand("hazard")

    def _rand(self, label: str):
        while True:
            x, y = random.randrange(GRID), random.randrange(GRID)
            if self.grid[x, y] == "empty":
                self.grid[x, y] = label
                break

    def cell(self, pos: Tuple[int, int]) -> str:
        x, y = pos
        return self.grid[x % GRID, y % GRID]

    def remove_food(self, pos: Tuple[int, int]):
        self.grid[pos] = "empty"

    def nearest_food_distance(self, pos: Tuple[int, int]) -> int:
        fx, fy = np.where(self.grid == "food")
        if fx.size == 0:
            return GRID
        dists = np.abs(fx - pos[0]) + np.abs(fy - pos[1])
        return int(dists.min())

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

# ───────────────────── Streamlit app ─────────────────────
st.set_page_config(
    page_title="Adaptive Hopfield Agent", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .cell-grid {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 10px;
    }
    .header {
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        color: #555;
        font-size: 18px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='header'>Adaptive Hopfield Agent</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>An exploratory agent that learns from experience using Hopfield networks</p>", unsafe_allow_html=True)

# Initialize state
if "world" not in st.session_state:
    # initialise world & agent (try loading state on agent)
    st.session_state.world = World()
    st.session_state.agent = Agent(st.session_state.world)
    st.session_state.agent.load_state()  # harmless if file absent
    st.session_state.running = False
    st.session_state.speed = 0.15  # Default speed
    logging.info("Session initialised.")

world: World = st.session_state.world
agent: Agent = st.session_state.agent

# --- sidebar controls ---
with st.sidebar:
    st.markdown("## Simulation Controls")
    col1, col2 = st.columns(2)
    
    if col1.button("▶️ Start", use_container_width=True):
        st.session_state.running = True
        logging.info("▶️ Simulation started.")
    if col2.button("⏸️ Pause", use_container_width=True):
        st.session_state.running = False
        logging.info("⏸️ Simulation paused.")
    
    st.session_state.speed = st.slider(
        "Simulation Speed", 
        min_value=0.01, 
        max_value=0.5, 
        value=st.session_state.speed,
        step=0.01,
        format="%.2f"
    )
    
    if st.button("🔄 Reset Simulation", use_container_width=True):
        # remove saved state and rebuild everything
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
            logging.info("🗑️ Saved state cleared.")
        st.session_state.world = World()
        st.session_state.agent = Agent(st.session_state.world)
        st.session_state.running = False
        st.rerun()
    
    # Display cell type experiences
    st.markdown("## Cell Type Learning")
    exp_data = []
    for cell_type, data in agent.cell_experience.items():
        exp_data.append({
            "Type": cell_type.capitalize(),
            "Reward": f"{data['reward']:.2f}",
            "Visits": data['visits']
        })
    
    st.dataframe(
        pd.DataFrame(exp_data),
        hide_index=True,
        use_container_width=True
    )
    
    # Show agent stats
    st.markdown("## Agent Stats")
    st.markdown(f"**Tick Count:** {agent.tick_count}")
    st.markdown(f"**Last Action:** {agent.last_action}")
    st.markdown(f"**Last Reward:** {agent.last_reward:.2f}")

# Main content area
main_col1, main_col2 = st.columns([2, 1])

with main_col1:
    # --- grid rendering with improved visuals ---
    color_map = {
        "home": [0, 0.8, 0],      # Green
        "food": [1, 0.8, 0],      # Yellow
        "hazard": [0.9, 0, 0],    # Red
        "empty": [0.9, 0.9, 0.9], # Light Gray
    }

    rgb = np.zeros((GRID, GRID, 3))
    for i, j in itertools.product(range(GRID), range(GRID)):
        rgb[i, j] = color_map[world.grid[i, j]]
    
    # Add agent position with different color if carrying food
    ax, ay = agent.pos
    rgb[ax, ay] = [0, 0.4, 0.9] if not agent.carrying else [0.8, 0, 0.8]
    
    # Create customized figure
    fig = px.imshow(rgb, aspect="equal")
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Add home position marker
    hx, hy = world.home
    fig.add_annotation(
        x=hy, 
        y=hx,
        text="🏠",
        showarrow=False,
        font=dict(size=16)
    )
    
    # Add agent marker
    fig.add_annotation(
        x=ay, 
        y=ax,
        text="🤖" if not agent.carrying else "🧠",
        showarrow=False,
        font=dict(size=16)
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with main_col2:
    # --- metrics with better visuals ---
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    
    # Energy bar
    energy_color = "green" if agent.energy > MAX_E * 0.5 else "orange" if agent.energy > MAX_E * 0.2 else "red"
    st.markdown(f"### Energy: {agent.energy:.0f}/{MAX_E}")
    st.progress(max(0.0, min(1.0, agent.energy / MAX_E)))
    
    # Hunger bar
    hunger_color = "green" if agent.hunger < MAX_H * 0.3 else "orange" if agent.hunger < MAX_H * 0.7 else "red"
    st.markdown(f"### Hunger: {agent.hunger}/{MAX_H}")
    st.progress(max(0.0, min(1.0, agent.hunger / MAX_H)))
    
    # Pain bar
    pain_color = "green" if agent.pain < MAX_P * 0.3 else "orange" if agent.pain < MAX_P * 0.7 else "red"
    st.markdown(f"### Pain: {agent.pain}/{MAX_P}")
    st.progress(max(0.0, min(1.0, agent.pain / MAX_P)))
    
    # Food stats
    st.markdown(f"### Food Stored: {agent.store}")
    st.markdown(f"### Carrying Food: {'Yes' if agent.carrying else 'No'}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# History charts
if agent.tick_count > 10:
    st.markdown("## Learning & Performance History")
    
    # Create history dataframe - ensure all arrays are the same length
    history_length = min(300, len(agent.history["energy"]))
    
    # Make sure rewards and actions match other metrics in length
    # If they're shorter (which can happen with loaded state), pad them
    rewards_len = len(agent.history["rewards"])
    if rewards_len < history_length and rewards_len > 0:
        padding_needed = history_length - rewards_len
        agent.history["rewards"] = [0] * padding_needed + agent.history["rewards"]
        
    # Only include rewards in dataframe if they exist
    if len(agent.history["rewards"]) >= history_length:
        df = pd.DataFrame({
            "Tick": range(agent.tick_count - history_length + 1, agent.tick_count + 1),
            "Energy": agent.history["energy"][-history_length:],
            "Hunger": agent.history["hunger"][-history_length:],
            "Pain": agent.history["pain"][-history_length:],
            "Food": agent.history["food_stored"][-history_length:],
            "Reward": agent.history["rewards"][-history_length:]
        })
    else:
        # Create DataFrame without rewards if they're not available
        df = pd.DataFrame({
            "Tick": range(agent.tick_count - history_length + 1, agent.tick_count + 1),
            "Energy": agent.history["energy"][-history_length:],
            "Hunger": agent.history["hunger"][-history_length:],
            "Pain": agent.history["pain"][-history_length:],
            "Food": agent.history["food_stored"][-history_length:]
        })
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Agent Status", "Rewards"),
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4]
    )
    
    # First plot: Agent Status
    fig.add_trace(
        go.Scatter(x=df["Tick"], y=df["Energy"], mode="lines", name="Energy", line=dict(color="blue")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["Tick"], y=df["Hunger"], mode="lines", name="Hunger", line=dict(color="orange")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["Tick"], y=df["Pain"], mode="lines", name="Pain", line=dict(color="red")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["Tick"], y=df["Food"], mode="lines", name="Food Stored", line=dict(color="green")),
        row=1, col=1
    )
    
    # Second plot: Rewards (only if rewards data exists)
    if "Reward" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["Tick"], y=df["Reward"], mode="lines", name="Reward", line=dict(color="purple")),
            row=2, col=1
        )
    else:
        # Add empty trace with a message when no reward data is available
        fig.add_annotation(
            text="Reward data will appear here after some actions",
            xref="paper", yref="paper",
            x=0.5, y=0.25,  # Position in the second subplot
            showarrow=False,
            font=dict(size=12)
        )
    
    # Update layout
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- autoplay tick ---
if st.session_state.running:
    agent.step()              # take one action & persist
    time.sleep(st.session_state.speed)  # wait based on speed setting
    st.rerun()
