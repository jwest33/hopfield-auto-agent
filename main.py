from __future__ import annotations
import itertools, random, time, logging, os, sys
from typing import Tuple

import numpy as np
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px

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
CARRY_HOME_W = 3.0              # incentive to bring food home
HOME_DIST_W = 2.0               # incentive to rest when fatigued
ENERGY_LOW_FRAC = 0.3
EXPLORATION_BONUS = 5.0         # reward for surprise when no food visible
HAZARD_PENALTY = 5.0
REST_PENALTY = 0.2
TICK_SEC = 0.15

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
        logging.info("Creating new world grid…")
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
        self.mem0.store(self.observe())
        logging.info("Agent initialised at home {}.".format(self.pos))

    # ───────────── persistence helpers ─────────────
    def save_state(self, path: str = STATE_FILE):
        """Persist internal memory & essential physiology."""
        np.savez_compressed(
            path,
            pos=self.pos,
            energy=self.energy,
            hunger=self.hunger,
            pain=self.pain,
            carrying=self.carrying,
            store=self.store,
            mem0_M=self.mem0.M,
            mem0_t=np.asarray(self.mem0.t, dtype=float),
            mem1_M=self.mem1.M,
            mem1_t=np.asarray(self.mem1.t, dtype=float),
        )

    def load_state(self, path: str = STATE_FILE):
        """Restore internal memory & physiology from *existing* file."""
        if not os.path.exists(path):
            logging.info("No previous state found - starting fresh.")
            return
        data = np.load(path, allow_pickle=True)
        self.pos = data["pos"].tolist()
        self.energy = float(data["energy"])
        self.hunger = float(data["hunger"])
        self.pain = float(data["pain"])
        self.carrying = bool(data["carrying"])
        self.store = int(data["store"])
        self.mem0.M = data["mem0_M"]
        self.mem0.t = data["mem0_t"].tolist()
        self.mem1.M = data["mem1_M"]
        self.mem1.t = data["mem1_t"].tolist()
        logging.info("Agent state loaded ← %s", path)

    # -------- perception --------
    def observe(self) -> np.ndarray:
        flat = self.w.grid.reshape(-1)
        return ENC.transform(flat[:, None]).flatten()

    # -------- planning --------
    def plan(self) -> str:
        hunger = self.hunger / MAX_H
        pain = self.pain / MAX_P
        energy_def = (MAX_E - self.energy) / MAX_E
        best, best_cost = "REST", float("inf")
        food_none = self.w.nearest_food_distance(tuple(self.pos)) == GRID
        for act, (dx, dy) in Agent.MOV.items():
            nxt = [(self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID]
            cell = self.w.cell(tuple(nxt))
            obs_nxt = self.observe() if act == "REST" else self._obs_after_move(nxt)
            cost = self.mem0.surprise(obs_nxt)
            if food_none:
                cost -= EXPLORATION_BONUS * (cost / SURPRISE_SCALE)
            cost += HUNGER_W * hunger + PAIN_W * pain + energy_def
            if self.carrying and act != "REST":
                cost += CARRY_COST
            if act == "REST":
                cost += REST_PENALTY * self.rest_streak
            if cell == "hazard":
                cost += HAZARD_PENALTY
            cost += self.w.nearest_food_distance(tuple(nxt)) / GRID
            if self.carrying:
                home_dist = abs(nxt[0] - self.w.home[0]) + abs(nxt[1] - self.w.home[1])
                cost += CARRY_HOME_W * (home_dist / GRID)
            if energy_def > ENERGY_LOW_FRAC:
                home_dist = abs(nxt[0] - self.w.home[0]) + abs(nxt[1] - self.w.home[1])
                cost += HOME_DIST_W * (home_dist / GRID)
            if np.isfinite(cost) and cost < best_cost:
                best, best_cost = act, cost
        logging.debug("Planned action %s with cost %.3f", best, best_cost)
        return best

    def _obs_after_move(self, nxt: list[int]) -> np.ndarray:
        cur = self.pos
        self.pos = nxt
        obs = self.observe()
        self.pos = cur
        return obs

    # -------- acting --------
    def step(self):
        act = self.plan()
        dx, dy = Agent.MOV[act]
        # movement or rest
        if act != "REST":
            self.pos = [(self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID]
            self.rest_streak = 0
            self.energy -= MOVE_COST
            if self.carrying:
                self.energy -= CARRY_COST
        else:
            self.rest_streak += 1

        # metabolism
        self.hunger = min(MAX_H, self.hunger + 1)
        self.pain = max(0, self.pain - PAIN_DECAY)
        self.energy -= self.hunger / MAX_H + self.pain / MAX_P

        cell = self.w.cell(tuple(self.pos))
        if cell == "hazard":
            self.pain = min(MAX_P, self.pain + PAIN_HIT)
        if cell == "food" and not self.carrying:
            self.carrying = True
            self.w.remove_food(tuple(self.pos))
        if cell == "home":
            if self.carrying and act != "REST":
                self.carrying = False
                self.store += 1
            if act == "REST":
                if self.carrying:
                    self.carrying = False
                    eat = True
                elif self.store > 0:
                    self.store -= 1
                    eat = True
                else:
                    eat = False
                if eat:
                    self.energy = min(MAX_E, self.energy + FOOD_E)
                    self.hunger = max(0, self.hunger - FOOD_S)

        # learning
        obs = self.observe()
        self.mem0.store(obs)
        if len(self.trace) >= SEQ_LEN - 1:
            seq = np.concatenate(self.trace[-(SEQ_LEN - 1):] + [obs])
            self.mem1.store(seq)
        self.trace.append(obs)

        # persistence
        self.save_state()
        logging.debug("Tick complete - pos %s, energy %.1f, hunger %d, pain %d", self.pos, self.energy, self.hunger, self.pain)

# ───────────────────── Streamlit app ─────────────────────
st.set_page_config("Hopfield forager", layout="wide")

if "world" not in st.session_state:
    # initialise world & agent (try loading state on agent)
    st.session_state.world = World()
    st.session_state.agent = Agent(st.session_state.world)
    st.session_state.agent.load_state()  # harmless if file absent
    st.session_state.running = False
    logging.info("Session initialised.")

world: World = st.session_state.world
agent: Agent = st.session_state.agent

# --- sidebar controls ---
left_btn, right_btn = st.sidebar.columns(2)
if left_btn.button("Start"):
    st.session_state.running = True
    logging.info("Simulation started.")
if right_btn.button("Stop"):
    st.session_state.running = False
    logging.info("Simulation paused.")
if st.sidebar.button("Reset"):
    # remove saved state and rebuild everything
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        logging.info("Saved state cleared.")
    st.session_state.world = World()
    st.session_state.agent = Agent(st.session_state.world)
    st.session_state.running = False
    st.rerun()

# --- grid rendering ---
color_map = {
    "home": [0, 1, 0],
    "food": [1, 1, 0],
    "hazard": [1, 0, 0],
    "empty": [0.85, 0.85, 0.85],
}

rgb = np.zeros((GRID, GRID, 3))
for i, j in itertools.product(range(GRID), range(GRID)):
    rgb[i, j] = color_map[world.grid[i, j]]
ax, ay = agent.pos  # avoid clobbering plotly.express alias
rgb[ax, ay] = [0, 0, 1] if not agent.carrying else [0.5, 0, 1]

fig = px.imshow(rgb, aspect="equal")
fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
st.plotly_chart(fig, use_container_width=True)

# --- metrics ---
left_m, right_m = st.columns(2)
left_m.metric("Energy", f"{agent.energy:.0f}/{MAX_E}")
left_m.metric("Hunger", f"{agent.hunger}/{MAX_H}")
left_m.metric("Pain", f"{agent.pain}/{MAX_P}")
right_m.metric("Food stored", str(agent.store))
right_m.metric("Carrying", "Yes" if agent.carrying else "No")

# --- autoplay tick ---
if st.session_state.running:
    agent.step()              # take one action & persist
    time.sleep(TICK_SEC)      # wait a bit so UI can paint
    st.rerun()
