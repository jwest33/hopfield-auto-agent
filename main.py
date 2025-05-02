from __future__ import annotations
import itertools, time, logging, os, sys, pickle
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from module_agent import Agent, AgentPopulation
from module_new_world import World  # Updated import from module_new_world

# ───────────────────── configuration ─────────────────────
GRID = 40  # Updated to match module_new_world's default
OBS_DIM = GRID * GRID * 8  # Expanded observation dimension for richer world model
SEQ_LEN, SEQ_DIM = 5, OBS_DIM * 5
CAP_L0, CAP_L1 = 800, 1200

MAX_E, MAX_H, MAX_P = 100, 100, 100
MOVE_COST, CARRY_COST = 1, 1
FOOD_E, FOOD_S = 60, 50  # Energy and satiety from food
PAIN_HIT, PAIN_DECAY = 10, 1  # Pain from hazards
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
WORLD_FILE = "world.pkl"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S",
                    stream=sys.stdout)

# ───────────────────── Streamlit app ─────────────────────
st.set_page_config(
    page_title="Multi-Agent Emergent Communication", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI (unchanged)
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
    .agent-selection {
        margin-bottom: 15px;
        padding: 10px;
        background-color: #f0f0f0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load a saved world or create a new one
def load_or_create_world(grid_size=GRID):
    if os.path.exists(WORLD_FILE):
        try:
            with open(WORLD_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading world: {e}")
            logging.info("Creating new world due to load error")
    
    # If no file exists or loading failed, create a new world
    return World(grid_size)

# Header
st.markdown("<h1 class='header'>Multi-Agent Emergent Communication</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Agents that develop their own communication patterns through neural networks</p>", unsafe_allow_html=True)

# Initialize state
if "world" not in st.session_state:
    # Initialize world & population
    st.session_state.world = load_or_create_world(GRID)
    st.session_state.population = AgentPopulation(st.session_state.world, initial_pop=5)
    st.session_state.running = False
    st.session_state.speed = 0.15
    st.session_state.selected_agent_id = list(st.session_state.population.agents.keys())[0] if st.session_state.population.agents else None
    logging.info("Multi-agent session initialized.")

world: World = st.session_state.world
population: AgentPopulation = st.session_state.population

# Get selected agent (for metrics display)
selected_agent = None
if st.session_state.selected_agent_id and st.session_state.selected_agent_id in population.agents:
    selected_agent = population.agents[st.session_state.selected_agent_id]
else:
    # If selected agent doesn't exist (was removed), select first available
    if population.agents:
        st.session_state.selected_agent_id = list(population.agents.keys())[0]
        selected_agent = population.agents[st.session_state.selected_agent_id]

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
        # Reset everything
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
            logging.info("🗑️ Saved state cleared.")
        st.session_state.world = load_or_create_world(GRID)
        st.session_state.population = AgentPopulation(st.session_state.world, initial_pop=5)
        st.session_state.running = False
        st.session_state.selected_agent_id = list(st.session_state.population.agents.keys())[0] if st.session_state.population.agents else None
        st.rerun()
    
    # Population controls
    st.markdown("## Population Controls")
    col1, col2 = st.columns(2)
    
    if col1.button("➕ Add Agent", use_container_width=True):
        new_agent = population.add_agent()
        if new_agent:
            st.session_state.selected_agent_id = new_agent.id
    
    if len(population.agents) > 1 and selected_agent and col2.button("❌ Remove Selected", use_container_width=True):
        population.remove_agent(st.session_state.selected_agent_id)
        st.session_state.selected_agent_id = list(population.agents.keys())[0] if population.agents else None
    
    # Agent selection
    if population.agents:
        st.markdown("## Agent Selection")
        agent_options = {agent_id: f"Agent {agent_id.split('_')[1]}" for agent_id in population.agents.keys()}
        selected_agent_name = st.selectbox(
            "Select Agent to Monitor", 
            options=list(agent_options.keys()),
            format_func=lambda x: agent_options[x],
            index=list(agent_options.keys()).index(st.session_state.selected_agent_id) if st.session_state.selected_agent_id in agent_options else 0
        )
        if selected_agent_name != st.session_state.selected_agent_id:
            st.session_state.selected_agent_id = selected_agent_name
            st.rerun()
    
    # Communication stats
    st.markdown("## Communication System")
    st.markdown(f"**Active Signals:** {len(population.comm_system.active_signals)}")
    
    # Population overview
    st.markdown("## Population Overview")
    st.markdown(f"**Active Agents:** {len(population.agents)}")
    
    # Show the 3 most recent interactions if there are any
    if population.interactions:
        st.markdown("## Recent Interactions")
        recent = population.interactions[-3:]
        for interaction in reversed(recent):
            st.markdown(f"**{interaction['type'].capitalize()}** between {interaction['agents'][0]} and {interaction['agents'][1]}")
    
    # World info
    st.markdown("## World Information")
    if hasattr(world, 'time'):
        st.markdown(f"**Time of Day:** {'Day' if world.is_day else 'Night'}")
        st.markdown(f"**Weather:** {world.weather.capitalize()}")
        st.markdown(f"**Temperature:** {world.ambient_temperature:.1f}°C")

# Main content area
main_col1, main_col2 = st.columns([2, 1])

with main_col1:
    # --- Grid rendering with improved visuals for the new world system ---
    # Define material colors
    material_color_map = {
        "dirt": [0.82, 0.70, 0.55],      # Brown
        "stone": [0.66, 0.66, 0.66],     # Gray
        "rock": [0.50, 0.50, 0.50],      # Dark Gray
        "water": [0.12, 0.56, 1.0],      # Blue
        "wood": [0.13, 0.55, 0.13],      # Green
        "food": [0.86, 0.08, 0.24],      # Red
        "home": [1.0, 0.84, 0.0],        # Gold
        "empty": [0.9, 0.9, 0.9],        # Light Gray
    }
    
    rgb = np.zeros((GRID, GRID, 3))
    
    # Process the TerrainCell grid to a color grid
    for i, j in itertools.product(range(GRID), range(GRID)):
        material = world.cell((i, j)).material
        rgb[i, j] = material_color_map.get(material, [0.9, 0.9, 0.9])  # Default to light gray
    
    # Create figure
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
    
    # Add markers for all agents with different colors/emojis
    agent_emojis = ["🤖", "🧠", "👾", "🦾", "🦿", "👁️"]
    for i, (agent_id, agent) in enumerate(population.agents.items()):
        ax, ay = agent.pos
        emoji_idx = i % len(agent_emojis)
        emoji = agent_emojis[emoji_idx]
        
        # Highlight selected agent
        if agent_id == st.session_state.selected_agent_id:
            # Add highlight circle around selected agent
            fig.add_shape(
                type="circle",
                x0=ay-0.4, y0=ax-0.4,
                x1=ay+0.4, y1=ax+0.4,
                line=dict(color="blue", width=2),
                fillcolor="rgba(0,0,255,0.1)"
            )
        
        # Show different emoji if carrying food
        display_emoji = emoji if not agent.carrying else "🍎"
        
        fig.add_annotation(
            x=ay, 
            y=ax,
            text=display_emoji,
            showarrow=False,
            font=dict(size=16)
        )
    
    # Add markers for active signals
    for signal in population.comm_system.active_signals:
        sx, sy = signal["position"]
        fig.add_annotation(
            x=sy,
            y=sx,
            text="💬",
            showarrow=False,
            font=dict(size=14)
        )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with main_col2:
    if selected_agent:
        # --- Agent metrics with better visuals ---
        st.markdown(f"<div class='metric-card'><h3>Agent {st.session_state.selected_agent_id.split('_')[1]} Stats</h3>", unsafe_allow_html=True)
        
        # Energy bar
        energy_color = "green" if selected_agent.energy > MAX_E * 0.5 else "orange" if selected_agent.energy > MAX_E * 0.2 else "red"
        st.markdown(f"### Energy: {selected_agent.energy:.0f}/{MAX_E}")
        st.progress(max(0.0, min(1.0, selected_agent.energy / MAX_E)))
        
        # Hunger bar
        hunger_color = "green" if selected_agent.hunger < MAX_H * 0.3 else "orange" if selected_agent.hunger < MAX_H * 0.7 else "red"
        st.markdown(f"### Hunger: {selected_agent.hunger}/{MAX_H}")
        st.progress(max(0.0, min(1.0, selected_agent.hunger / MAX_H)))
        
        # Pain bar
        pain_color = "green" if selected_agent.pain < MAX_P * 0.3 else "orange" if selected_agent.pain < MAX_P * 0.7 else "red"
        st.markdown(f"### Pain: {selected_agent.pain}/{MAX_P}")
        st.progress(max(0.0, min(1.0, selected_agent.pain / MAX_P)))
        
        # Food stats
        st.markdown(f"### Food Stored: {selected_agent.store}")
        st.markdown(f"### Carrying Food: {'Yes' if selected_agent.carrying else 'No'}")
        
        # Communication stats
        st.markdown("### Communication")
        if selected_agent.last_signal is not None:
            st.markdown(f"Last Signal: {selected_agent.tick_count - selected_agent.last_signal_time} ticks ago")
        else:
            st.markdown("No recent signals")
        
        # Environment stats - new for module_new_world
        current_cell = world.cell(tuple(selected_agent.pos))
        st.markdown("### Environment")
        st.markdown(f"Terrain: {current_cell.material.title()}")
        st.markdown(f"Temperature: {current_cell.temperature:.1f}°C")
        st.markdown(f"Risk: {current_cell.local_risk * 100:.1f}%")
        st.markdown(f"Passable: {'Yes' if current_cell.passable else 'No'}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No agent selected. Add agents to the simulation.")

# History charts for selected agent
if selected_agent and selected_agent.tick_count > 10:
    st.markdown("## Agent History")
    
    # Create history dataframe - ensure all arrays are the same length
    history_length = min(300, len(selected_agent.history["energy"]))
    
    # Make sure rewards and actions match other metrics in length
    # If they're shorter (which can happen with loaded state), pad them
    rewards_len = len(selected_agent.history["rewards"])
    if rewards_len < history_length and rewards_len > 0:
        padding_needed = history_length - rewards_len
        selected_agent.history["rewards"] = [0] * padding_needed + selected_agent.history["rewards"]
        
    # Only include rewards in dataframe if they exist
    if len(selected_agent.history["rewards"]) >= history_length:
        df = pd.DataFrame({
            "Tick": range(selected_agent.tick_count - history_length + 1, selected_agent.tick_count + 1),
            "Energy": selected_agent.history["energy"][-history_length:],
            "Hunger": selected_agent.history["hunger"][-history_length:],
            "Pain": selected_agent.history["pain"][-history_length:],
            "Food": selected_agent.history["food_stored"][-history_length:],
            "Reward": selected_agent.history["rewards"][-history_length:]
        })
    else:
        # Create DataFrame without rewards if they're not available
        df = pd.DataFrame({
            "Tick": range(selected_agent.tick_count - history_length + 1, selected_agent.tick_count + 1),
            "Energy": selected_agent.history["energy"][-history_length:],
            "Hunger": selected_agent.history["hunger"][-history_length:],
            "Pain": selected_agent.history["pain"][-history_length:],
            "Food": selected_agent.history["food_stored"][-history_length:]
        })
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f"Agent {selected_agent.id.split('_')[1]} Status", "Rewards"),
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

# Population overview dashboard (new)
if len(population.agents) > 1:
    st.markdown("## Population Overview")
    
    # Create columns for population metrics
    metrics_cols = st.columns(5)
    
    # Calculate population averages
    avg_energy = sum(agent.energy for agent in population.agents.values()) / len(population.agents)
    avg_hunger = sum(agent.hunger for agent in population.agents.values()) / len(population.agents)
    avg_pain = sum(agent.pain for agent in population.agents.values()) / len(population.agents)
    total_food_stored = sum(agent.store for agent in population.agents.values())
    carrying_count = sum(1 for agent in population.agents.values() if agent.carrying)
    
    # Display metrics
    metrics_cols[0].metric("Avg Energy", f"{avg_energy:.1f}")
    metrics_cols[1].metric("Avg Hunger", f"{avg_hunger:.1f}")
    metrics_cols[2].metric("Avg Pain", f"{avg_pain:.1f}")
    metrics_cols[3].metric("Total Food Stored", total_food_stored)
    metrics_cols[4].metric("Agents Carrying Food", carrying_count)
    
    # Communication visualization (simplified)
    if population.comm_system.active_signals:
        st.markdown("### Active Communications")
        
        # Get active signal data
        signals_df = pd.DataFrame([
            {
                "Sender": signal["sender"].split("_")[1],
                "Position X": signal["position"][0],
                "Position Y": signal["position"][1],
                "Range": signal["range"]
            }
            for signal in population.comm_system.active_signals
        ])
        
        # Display active signals
        st.dataframe(signals_df, hide_index=True)

# --- autoplay tick ---
if st.session_state.running:
    # Step world and all agents
    world.step()
    population.step_all()
    time.sleep(st.session_state.speed)
    st.rerun()
