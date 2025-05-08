from __future__ import annotations
import itertools, time, logging, os, sys
from typing import List, Dict, Any
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from module_agent import Agent, MAX_E, MAX_H, MAX_P
from module_world import World

GRID = 25

st.set_page_config(
    page_title="Agent Performance Dashboard",
    layout="wide",
    initial_sidebar_state="auto",
)

# --- Sidebar Controls ---
with st.sidebar:
    st.title("Simulation Controls")
    if st.button("Start"):
        st.session_state.running = True
    if st.button("Pause"):
        st.session_state.running = False
    st.session_state.speed = st.slider(
        "Speed", 0.01, 1.0, st.session_state.get('speed', 0.1), 0.01
    )
    if st.button("Reset"):
        st.session_state.clear()
        st.rerun()

# --- Initialize World & Agents ---
if 'world' not in st.session_state:
    st.session_state.world = World(grid_size=GRID)
    st.session_state.agents = [Agent(st.session_state.world, f"agent_{i+1}") for i in range(2)]
    for ag in st.session_state.agents:
        ag.load_state(path=f"{ag.agent_id}_state.npz")
    st.session_state.running = False
    st.session_state.speed = 0.1

world: World = st.session_state.world
agents: List[Agent] = st.session_state.agents

# --- Comparative Metrics ---
st.header("Comparative Agent Metrics")
summary: Dict[str, Any] = {
    ag.agent_id: {
        'Food Collected': ag.store,
        'Average Reward': float(np.mean(ag.history['rewards'])) if ag.history['rewards'] else 0.0,
        'Hazards Stepped': sum(1 for v in ag.known_map.values() if v == 'hazard'),
        'Steps Taken': ag.tick_count,
    }
    for ag in agents
}

df_summary = pd.DataFrame(summary).T
st.dataframe(df_summary, use_container_width=True)

# --- Layout: World Grid & Agent Status ---
col1, col2 = st.columns([2, 1])

# World Grid Visualization
with col1:
    st.header("World Grid")
    grid_rgb = np.zeros((GRID, GRID, 3))
    palette = {'home': [0, 1, 0], 'food': [1, 1, 0], 'hazard': [1, 0, 0], 'empty': [0.9, 0.9, 0.9]}
    for i, j in itertools.product(range(GRID), range(GRID)):
        grid_rgb[i, j] = palette.get(world.grid[i, j], [0.9, 0.9, 0.9])
    fig = go.Figure(go.Image(z=(grid_rgb * 255).astype(np.uint8)))
    for idx, ag in enumerate(agents):
        x, y = ag.pos
        fig.add_trace(go.Scatter(x=[y], y=[x], mode="markers+text",
                                 marker=dict(size=20, symbol="circle"),
                                 text=[str(idx+1)], textfont=dict(color="white", size=12),
                                 showlegend=False))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), yaxis={'autorange': 'reversed'})
    st.plotly_chart(fig, use_container_width=True, height=500)

# Agent Status Side-by-Side
with col2:
    st.header("Agent Status")
    agent_cols = st.columns(len(agents))
    for idx, (ag, agent_col) in enumerate(zip(agents, agent_cols)):
        with agent_col:
            st.subheader(f"Agent {idx+1}")
            st.metric("Energy", f"{ag.energy:.0f}/{MAX_E}")
            st.metric("Hunger", f"{ag.hunger:.0f}/{MAX_H}")
            st.metric("Pain", f"{ag.pain:.0f}/{MAX_P}")
            st.metric("Carrying Food", "Yes" if ag.carrying else "No")

st.header("Performance Over Time")
history_records: Dict[str, List[Any]] = {'Tick': [], 'Agent': [], 'Energy': [], 'Hunger': [], 'Pain': [], 'Reward': []}
for ag in agents:
    for i in range(len(ag.history['energy'])):
        history_records['Tick'].append(i + 1)
        history_records['Agent'].append(ag.agent_id)
        history_records['Energy'].append(ag.history['energy'][i])
        history_records['Hunger'].append(ag.history['hunger'][i])
        history_records['Pain'].append(ag.history['pain'][i])
        history_records['Reward'].append(ag.history['rewards'][i] if i < len(ag.history['rewards']) else 0.0)

df_hist = pd.DataFrame(history_records)
fig_e = px.line(df_hist, x='Tick', y='Energy', color='Agent', title='Energy Over Time')
fig_r = px.line(df_hist, x='Tick', y='Reward', color='Agent', title='Reward Over Time')
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.plotly_chart(fig_e, use_container_width=True)
with chart_col2:
    st.plotly_chart(fig_r, use_container_width=True)

# --- Main Loop ---
if st.session_state.running:
    for ag in agents:
        ag.step()
    time.sleep(st.session_state.speed)
    st.rerun()
