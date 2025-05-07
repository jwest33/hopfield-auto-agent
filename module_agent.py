from __future__ import annotations
import itertools, random, time, logging, os, sys
from typing import Tuple, Dict, Any
import numpy as np
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
from module_hopfield import Hopfield
from module_world import World
from module_comm import default_comm as COMM_CHANNEL, VQ, Message

# ───────────────────── configuration ─────────────────────
GRID = 25
OBS_DIM = GRID * GRID * 4
SEQ_LEN = 5
SEQ_DIM = OBS_DIM * SEQ_LEN  # total obs dims in a segment
SEQ_DIM_REWARD = SEQ_DIM + 1 # +1 slot for the reward scalar
CAP_L0, CAP_L1 = 800, 1200

# Communication parameters
N_SYMBOLS = 32           # size of discrete vocabulary
SEQ_LEN = 5              # length of trace to package
REWARD_THRESH = 1.0      # packaging trigger threshold for reward
SURPRISE_THRESH = 0.5    # packaging trigger threshold for surprise

# Agent physiology parameters 
MAX_E, MAX_H, MAX_P = 100, 100, 100
MOVE_COST, CARRY_COST = 1, 1

# Energy recovered per tick when resting (off-grid)
REST_ENERGY_RECOVERY = MOVE_COST

FOOD_E, FOOD_S = 40, 40
PAIN_HIT, PAIN_DECAY = 5, 5 

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

    def __init__(self, w: World, agent_id: str):
        self.agent_id = agent_id
        self.w = w
        self.pos = list(w.home)
        self.energy, self.hunger, self.pain = MAX_E, 0, 0

        self.carrying, self.store = False, 0
        self.rest_streak = 0
        self.mem0 = Hopfield(OBS_DIM, CAP_L0)
        self.mem1 = Hopfield(SEQ_DIM_REWARD, CAP_L1)
        self.trace: list[np.ndarray] = []

        # Tracking
        self.tick_count = 0
        self.last_reward = 0.0
        self.last_surprise = 0.0
        self.last_action = None

        # Communication setup
        self.comm = COMM_CHANNEL
        self.vq = VQ(n_clusters=N_SYMBOLS, dim=SEQ_DIM_REWARD)
        self.symbol_labels: Dict[int, str] = {i: f"Cluster {i}" for i in range(N_SYMBOLS)}
        self.comm_log: list[tuple[int, int, str]] = []  # (tick, symbol, label)

        self.comm.register(self)

        # Memory & learning
        self.cell_experience = {t: {"reward": 0, "visits": 0, "last_visit": 0}
                                 for t in ["home", "food", "hazard", "empty"]}
        self.q_values: Dict[str, Dict[str, float]] = {}

        # History
        self.history = {"energy": [MAX_E], "hunger": [0], "pain": [0],
                        "food_stored": [0], "actions": [], "rewards": [], "surprise": [0.0]}

        # Seed memory
        init_obs = self.observe()
        self.mem0.store(init_obs)
        self.trace.append(init_obs)
        logging.info(f"Agent initialized at home {self.pos}.")

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
            mem1_M=self.mem1.M,
            cell_experience=np.array([self.cell_experience], dtype=object),
            q_values=np.array([self.q_values], dtype=object),
            history=np.array([self.history], dtype=object)
        )

    def load_state(self, path: str = STATE_FILE):
        """Restore internal memory & physiology from existing file if present."""
        if not os.path.exists(path):
            logging.info("No previous state found – starting fresh.")
            return
        try:
            data = np.load(path, allow_pickle=True)
            self.pos = data["pos"].tolist()
            self.energy = float(data["energy"])
            self.hunger = float(data["hunger"])
            self.pain = float(data["pain"])
            self.carrying = bool(data["carrying"])
            self.store = int(data["store"])
            if "tick_count" in data:
                self.tick_count = int(data["tick_count"])
            self.mem0.M = data["mem0_M"]
            self.mem1.M = data["mem1_M"]
            if "cell_experience" in data:
                self.cell_experience = data["cell_experience"].item()
            if "q_values" in data:
                self.q_values = data["q_values"].item()
            if "history" in data:
                self.history = data["history"].item()
            logging.info("Agent state loaded from %s", path)
        except Exception as e:
            logging.error(f"Error loading state: {e}")
            logging.info("Starting fresh due to load error")

    def update_experience(self, cell_type: str, reward: float):
        exp = self.cell_experience[cell_type]
        exp["reward"] = reward if exp["visits"] == 0 else (exp["reward"] * 0.9 + reward * 0.1)
        exp["visits"] += 1
        exp["last_visit"] = self.tick_count

    def decay_experiences(self):
        for cell_type, exp in self.cell_experience.items():
            if exp["visits"] > 0:
                exp["reward"] *= EXPERIENCE_DECAY

    def get_state_key(self, pos: list[int], carrying: bool = None) -> str:
        carrying_val = self.carrying if carrying is None else carrying
        return f"{tuple(pos)}|{carrying_val}"

    def get_q_value(self, state_key: str, action: str) -> float:
        if state_key not in self.q_values:
            self.q_values[state_key] = {act: 0.0 for act in self.MOV}
        return self.q_values[state_key].get(action, 0.0)

    def update_q_value(self, state_key: str, action: str, reward: float, next_state_key: str):
        if state_key not in self.q_values:
            self.q_values[state_key] = {act: 0.0 for act in self.MOV}
        if next_state_key not in self.q_values:
            self.q_values[next_state_key] = {act: 0.0 for act in self.MOV}
        next_max = max(self.q_values[next_state_key].values())
        old_q = self.q_values[state_key][action]
        self.q_values[state_key][action] = old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max - old_q)

    def observe(self) -> np.ndarray:
        flat = self.w.grid.reshape(-1)
        return ENC.transform(flat[:, None]).flatten()

    def plan(self) -> str:
        hunger = self.hunger / MAX_H
        pain = self.pain / MAX_P
        energy_def = (MAX_E - self.energy) / MAX_E
        state_k = self.get_state_key(self.pos)
        food_none = self.w.nearest_food_distance(tuple(self.pos)) == GRID

        if self.energy < MAX_E * 0.15:
            cell = self.w.cell(tuple(self.pos))
            if cell == "home":
                logging.info("Critical energy: resting at home")
                return "REST"
            dx = self.w.home[0] - self.pos[0]
            dy = self.w.home[1] - self.pos[1]
            if abs(dx) > abs(dy):
                return "S" if dx > 0 else "N"
            return "E" if dy > 0 else "W"

        epsilon = max(0.1, 1.0/(1 + self.tick_count/1000))
        if random.random() < epsilon:
            return random.choice(list(self.MOV))

        best_act, best_val = "REST", float('-inf')
        for act, (dx, dy) in self.MOV.items():
            nxt = [(self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID]
            cell = self.w.cell(tuple(nxt))
            val = self.get_q_value(state_k, act)
            exp = self.cell_experience[cell]
            if exp["visits"] > 0:
                val += exp["reward"]
            if exp["visits"] < 10:
                val += 2.0 / (exp["visits"] + 1)
            obs_n = self.observe() if act == "REST" else self._obs_after_move(nxt)
            surpr = self.mem0.surprise(obs_n)
            if food_none:
                val += EXPLORATION_BONUS * (surpr / SURPRISE_SCALE)
            val -= HUNGER_W*hunger + PAIN_W*pain + energy_def*2.0
            if self.carrying and act != "REST":
                val -= CARRY_COST
            if act == "REST":
                val -= REST_PENALTY * self.rest_streak
                if self.energy < MAX_E*0.4:
                    bonus = (MAX_E*0.4 - self.energy)/MAX_E*10
                    val += bonus * (2 if cell == "home" else 1)
            if self.carrying:
                dist = abs(nxt[0]-self.w.home[0]) + abs(nxt[1]-self.w.home[1])
                val -= CARRY_HOME_W * (dist / GRID)
            if energy_def > ENERGY_LOW_FRAC:
                dist = abs(nxt[0]-self.w.home[0]) + abs(nxt[1]-self.w.home[1])
                val -= HOME_DIST_W * (dist/GRID) * (1 + energy_def*2)
            if val > best_val:
                best_act, best_val = act, val
        if self.energy < MAX_E*0.25 and best_act != "REST" and self.w.cell(tuple(self.pos))=="home":
            logging.info(f"Low energy override: REST")
            return "REST"
        return best_act

    def _obs_after_move(self, nxt: list[int]) -> np.ndarray:
        cur = self.pos
        self.pos = nxt
        v = self.observe()
        self.pos = cur
        return v

    def step(self):
        """Take one step in the environment and learn."""
        self.tick_count += 1
        prev_pos, prev_carry = self.pos.copy(), self.carrying
        prev_en, prev_p, prev_h = self.energy, self.pain, self.hunger
        prev_key = self.get_state_key(prev_pos, prev_carry)

        # Receive any queries
        self._receive_messages()

        act = self.plan()
        self.last_action = act
        dx, dy = self.MOV[act]

        # Move or rest
        if act != "REST":
            self.pos = [(self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID]
            self.rest_streak = 0
            self.energy -= MOVE_COST
            if self.carrying:
                self.energy -= CARRY_COST
        else:
            self.rest_streak += 1
            rec = REST_ENERGY_RECOVERY * (2 if self.w.cell(tuple(self.pos)) == "home" else 1)
            self.energy = min(MAX_E, self.energy + rec)

        # Always increase hunger
        self.hunger = min(MAX_H, self.hunger + 1)

        # Interact with cell
        food_collected = False
        food_eaten = False
        curr = self.w.cell(tuple(self.pos))
        if curr == "hazard":
            old_pain = self.pain
            self.pain = min(MAX_P, self.pain + PAIN_HIT)
            self.energy = max(0, self.energy - (self.pain - old_pain))
        if curr == "food" and not self.carrying:
            self.carrying = True
            food_collected = True
            self.w.remove_food(tuple(self.pos))
        if curr == "home":
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

        # Compute reward
        de = self.energy - prev_en
        dp = self.pain - prev_p
        dh = self.hunger - prev_h
        reward = de - dp - dh + (10 if food_collected else 0) + (20 if food_eaten else 0)
        self.last_reward = reward

        # Surprise and memory
        obs = self.observe()
        self.last_surprise = self.mem0.surprise(obs)
        self.mem0.store(obs)
        if len(self.trace) >= SEQ_LEN:
            segment = np.concatenate(self.trace[-SEQ_LEN+1:] + [obs] + [np.array([self.last_reward])])
            self.mem1.store(segment)
        self.trace.append(obs)

        # Update learning
        self.update_experience(curr, reward)
        next_key = self.get_state_key(self.pos, self.carrying)
        self.update_q_value(prev_key, act, reward, next_key)
        if self.tick_count % 100 == 0:
            self.decay_experiences()

        # Log history and persist
        self.history["energy"].append(self.energy)
        self.history["hunger"].append(self.hunger)
        self.history["pain"].append(self.pain)
        self.history["food_stored"].append(self.store)
        self.history["actions"].append(act)
        self.history["rewards"].append(reward)
        self.history["surprise"].append(self.last_surprise)
        self.save_state(path=f"{self.agent_id}_state.npz")

        logging.debug(f"Tick {self.tick_count} – pos {self.pos}, energy {self.energy}, hunger {self.hunger}, pain {self.pain}, reward {reward}")

    def _receive_messages(self):
        msgs = self.comm.receive_all()
        for msg in msgs:
            if msg.data.get("query"):
                self.receive(msg)

    def receive(self, msg: Message):
        """Handle incoming query and respond."""
        logging.info(f"Agent {self.agent_id} received query from {msg.sender} at tick {self.tick_count}.")
        self._respond_to_query(msg.sender)

    def query_agent(self, target_id: str):
        """Broadcast a query to other agents."""
        msg = Message(sender=self.agent_id, symbol=-1, data={"query": True})
        self.comm.broadcast(msg)

    def _respond_to_query(self, requester_id: str):
        """Respond with last known cluster symbol and label."""
        if not self.comm_log:
            return
        tick, symbol, label = self.comm_log[-1]
        msg = Message(sender=self.agent_id, symbol=symbol,
                      data={"label": label, "response_to": requester_id})
        self.comm.broadcast(msg)
