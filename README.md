# Hop To It: Experience-Based Grid World Learning

## Overview

This project implements an autonomous agent that navigates a grid-based world using Hopfield networks for memory and experience-based learning for decision-making. The agent balances exploration, resource gathering, and self-preservation while adapting to its environment through reinforcement learning rather than hardcoded rules.

## Features

- **Experience-Based Learning**: Agent learns to avoid hazards and seek rewards through experience rather than hardcoded rules
- **Reinforcement Learning**: Uses Q-learning to update the agent's understanding of its environment
- **Persistent Memory**: Agent saves its neural networks, learning experiences, and physiological state between runs
- **Fatigue-Aware**: Agent considers energy, hunger, and pain in its decision-making with exponential penalties for critical states
- **Exploratory Behavior**: Uses surprise from Hopfield networks and epsilon-greedy exploration to encourage discovery
- **Enhanced Home Benefits**: Special recovery mechanics when resting at home provide adaptive safe-haven behavior
- **Interactive Visualization**: Rich Streamlit interface with real-time metrics and performance history

## Technical Architecture

### Learning System

The agent implements a multi-layered learning approach:

1. **Experience Memory**: Tracks rewards associated with different cell types
   ```python
   self.cell_experience = {
       "home": {"reward": 0, "visits": 0, "last_visit": 0},
       "food": {"reward": 0, "visits": 0, "last_visit": 0},
       "hazard": {"reward": 0, "visits": 0, "last_visit": 0},
       "empty": {"reward": 0, "visits": 0, "last_visit": 0}
   }
   ```

2. **Q-Learning**: Maintains state-action values that are updated using:
   ```python
   new_q = old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max_q - old_q)
   ```

3. **Exploration Strategy**: Uses epsilon-greedy approach that decreases over time:
   ```python
   epsilon = max(0.1, 1.0 / (1 + self.tick_count / 1000))
   ```

4. **Exponential Reward System**: Creates stronger learning signals for critical states:
   ```python
   energy_penalty = -10.0 * energy_critical * energy_critical if energy_critical > 0.5 else 0
   ```

### Agent Decision-Making

The agent uses a sophisticated planning system that combines learned values with heuristics:

1. Considers all possible moves (North, South, East, West, or Rest)
2. Calculates a value for each action based on:
   - Q-values from past experience
   - Experience-based rewards for different cell types
   - Exploration bonuses for less-visited cell types
   - Memory-based surprise from Hopfield networks
   - Current physiological needs (hunger, pain, energy)
   - Goal-directed behavior (returning home when carrying food or low on energy)
3. Selects the highest-value action
4. Updates learning based on observed outcomes

### Neural Foundations: Hopfield Networks

At the core of the agent's memory are two Hopfield networks:

- **Primary Memory (mem0)**: Stores observations of the immediate environment
- **Sequential Memory (mem1)**: Stores sequences of observations to capture temporal patterns

These networks implement a modern continuous Hopfield network with a softmax-based update rule:

```python
logits = self.M @ y - (self.M @ y).max()
p = np.exp(logits)
y = (p / s) @ self.M
```

This approach allows for a form of content-addressable memory that drives the agent's curiosity and novelty detection.

## Implementation Details

- **Grid Representation**: 25×25 grid with color-coded cell types (home, food, hazard, empty)
- **One-Hot Encoding**: Converts categorical cell types to vector representation
- **Streamlit Interface**: Advanced visualization with metrics, history charts, and interactive controls
- **Numpy-based Persistence**: Compressed storage of agent memory, learning, and state
- **Analytics Tools**: Standalone analyzer script for investigating agent learning patterns

## Usage

### Running the Simulation

```bash
streamlit run main.py
```

### Control Options
- **Start/Pause**: Begin or pause the simulation
- **Speed Slider**: Adjust simulation speed
- **Reset**: Clear state and reinitialize the world

### Analyzing Agent State

Use the agent state viewer to analyze learning progress:

```bash
python agent_state_viewer.py                # Basic view
python agent_state_viewer.py --learning     # Detailed learning analysis 
python agent_state_viewer.py --plot         # Visualize memory patterns
python agent_state_viewer.py --export       # Export data for further analysis
```

## Project Structure

```
.
├── main.py                # Main application with agent and world logic
├── agent_state_viewer.py  # Analysis tool for inspecting agent state
├── agent_state.npz        # Persisted agent state (created at runtime)
└── README.md              # This documentation file
```

## Future Work

Potential extensions:
- Curiosity-driven exploration with intrinsic motivation
- Dynamic environment with resource regeneration
- Multi-agent interactions and emergent social behaviors
- Meta-learning to adapt learning parameters over time
- Integration with larger environment scales and more diverse cell types
