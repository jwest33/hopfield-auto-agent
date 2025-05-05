from sklearn.preprocessing import OneHotEncoder
import numpy as np

GRID = 40
FPS = 30
CELL_SIZE = 18
PANEL_WIDTH = 500  # Increased panel width
WINDOW_WIDTH = GRID * CELL_SIZE + PANEL_WIDTH
WINDOW_HEIGHT = 800  # Increased window height
CLOCK_SPEED = 10  # Simulation ticks per second at normal speed

# File paths
STATE_FILE = "agent_state.npz"
WORLD_FILE = "world.pkl"
SAVE_DIR = "saves"

# Material colors
MATERIAL_COLORS = {
    "dirt": (210, 180, 140),
    "stone": (169, 169, 169),
    "rock": (128, 128, 128),
    "water": (30, 144, 255),
    "wood": (34, 139, 34),
    "food": (220, 20, 60),
    "home": (255, 215, 0),
    "empty": (230, 230, 230),
}

# UI colors
COLOR_BG = (30, 30, 30)
COLOR_PANEL = (50, 50, 50)
COLOR_TEXT = (240, 240, 240)
COLOR_TEXT_HEADER = (255, 255, 255)
COLOR_BUTTON = (80, 80, 80)
COLOR_BUTTON_HOVER = (100, 100, 100)
COLOR_BUTTON_ACTIVE = (120, 120, 120)

# Button states
BUTTON_NORMAL = 0
BUTTON_HOVER = 1
BUTTON_PRESSED = 2
OBS_DIM = GRID * GRID * 8  # Expanded for more terrain features
SEQ_LEN, SEQ_DIM = 5, OBS_DIM * 5
CAP_L0, CAP_L1 = 800, 1200

MAX_E, MAX_H, MAX_P = 100, 100, 100
MOVE_COST, CARRY_COST = 1, 1
FOOD_E, FOOD_S = 40, 40
PAIN_HIT, PAIN_DECAY = 25, 1
REST_ENERGY_RECOVERY = 10
HOME_ENERGY_RECOVERY = 20

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
