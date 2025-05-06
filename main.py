from __future__ import annotations
import itertools, time, logging, os, sys
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from module_agent import Agent
from module_world import World

# ───────────────────── configuration ─────────────────────
GRID = 25
OBS_DIM = GRID * GRID * 4
SEQ_LEN, SEQ_DIM = 5, OBS_DIM * 5
CAP_L0, CAP_L1 = 800, 1200

MAX_E, MAX_H, MAX_P = 100, 100, 100
MOVE_COST, CARRY_COST = 1, 1
FOOD_E, FOOD_S = 60, 50  # Increased energy and satiety from food
PAIN_HIT, PAIN_DECAY = 10, 1  # Reduced pain from hazards
HOME_ENERGY_RECOVERY = 5  # Additional energy recovery at home when resting
HOME_HUNGER_RECOVERY = 5  # Additional hunger reduction at home when resting

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
if "world" not in st.session_state or "agents" not in st.session_state:
    # create/reset the world
    st.session_state.world = World(grid_size=GRID)

    # create your agents
    st.session_state.agents = [
        Agent(st.session_state.world, agent_id="agent_1"),
        Agent(st.session_state.world, agent_id="agent_2"),
    ]

    # load each from its own file so they don't collide
    for agent in st.session_state.agents:
        agent.load_state(path=f"{agent.agent_id}_state.npz")

    st.session_state.running = False
    st.session_state.speed = 0.15

# now grab them locally
world = st.session_state.world
agents = st.session_state.agents

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
        # delete each agent's state file
        for agent in st.session_state.agents:
            fn = f"{agent.agent_id}_state.npz"
            if os.path.exists(fn):
                os.remove(fn)

        # re-init exactly the same way as above
        st.session_state.world = World(grid_size=GRID)
        st.session_state.agents = [
            Agent(st.session_state.world, agent_id="agent_1"),
            Agent(st.session_state.world, agent_id="agent_2"),
        ]
        for agent in st.session_state.agents:
            agent.load_state(path=f"{agent.agent_id}_state.npz")

        st.session_state.running = False
        st.rerun()
    for agent in st.session_state.agents:
        st.markdown(f"### 📊 Stats for {agent.agent_id}")
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

        # New: Memory & Communication
        st.markdown("## Agent Memory")
        # Memory usage
        mem0_used = agent.mem0.M.shape[0]
        mem1_used = agent.mem1.M.shape[0]
        st.metric("Hopfield Mem0 Usage", f"{mem0_used}/{CAP_L0}")
        st.metric("Hopfield Mem1 Usage", f"{mem1_used}/{CAP_L1}")
        # Surprise
        st.metric("Last Surprise", f"{agent.last_surprise:.2f}")

        # Symbol counts
        #st.markdown("### Communication Symbol Counts")
        #df_syms = pd.DataFrame.from_dict(agent.symbol_counts, orient='index', columns=["Count"] )
        #st.bar_chart(df_syms)

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
    
    colors = [
        [0, 0.4, 0.9],   # agent_1 color
        [0.8, 0, 0.8],   # agent_2 color
    ]
    for idx, agent in enumerate(st.session_state.agents):
        x, y = agent.pos
        rgb[x, y] = colors[idx % len(colors)]
        fig.add_annotation(
            x=y, y=x,
            text="🤖",
            showarrow=False,
            font=dict(size=16)
        )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with main_col2:
    
    for agent in st.session_state.agents:
    # --- metrics with better visuals ---
        # Energy bar
        st.markdown(f"### {agent.agent_id.capitalize()}:")
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
    for agent in st.session_state.agents:
        agent.step()  # each agent takes its turn
    time.sleep(st.session_state.speed)
    st.rerun()
