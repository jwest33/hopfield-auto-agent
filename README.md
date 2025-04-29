# Surprise World: Exploratory & Fatigue-Aware Hopfield Agent

## Overview

This project implements an autonomous agent that navigates a grid-based world using Hopfield networks for memory and decision-making. The agent must balance exploration, resource gathering, and self-preservation while navigating a world containing food, hazards, and a home base.

## Features

- **Persistence**: Agent saves its internal memory and physiological state between runs
- **Fatigue-Aware**: Agent considers energy, hunger, and pain in its decision-making
- **Exploratory Behavior**: Uses surprise from Hopfield networks to encourage exploration
- **Adaptive Planning**: Dynamically adjusts priorities based on current needs and environment

## Technical Architecture

### Agent Decision-Making

The agent uses a sophisticated cost-based planning system that evaluates multiple possible actions and selects the one with the lowest associated cost. This process occurs in the `plan()` method where the agent:

1. Considers all possible moves (North, South, East, West, or Rest)
2. Calculates a cost for each action based on:
   - Memory-based surprise value
   - Current physiological needs (hunger, pain, energy)
   - Environmental factors (distance to food/home, hazards)
   - Context-dependent costs (carrying food, energy levels)
3. Selects and executes the lowest-cost action
4. Updates internal state and learns from new observations

### Mathematical Foundations

#### Hopfield Networks

At the core of the agent's cognition are two Hopfield networks:

- **Primary Memory (mem0)**: Stores observations of the immediate environment
- **Sequential Memory (mem1)**: Stores sequences of observations to capture temporal patterns

These networks implement a modern continuous Hopfield network with a softmax-based update rule, using the following key operations:

- **Storage**: `store(v)` - Adds new patterns with timestamp-based priority
- **Recall**: `recall(v, it=3)` - Iteratively reconstructs patterns from memory
- **Surprise**: `surprise(v)` - Quantifies novelty as the distance between input and recall

The update formula uses a soft winner-takes-all approach:
```
logits = self.M @ y - (self.M @ y).max()
p = np.exp(logits)
y = (p / s) @ self.M
```

This approach allows for a form of content-addressable memory that drives the agent's curiosity and exploration behaviors.

#### Cost Function Components

The decision-making process is governed by a multi-factor cost function that balances:

```
cost = surprise + hunger_weight * hunger + pain_weight * pain + energy_deficit
       + context_specific_factors
```

Where context-specific factors include:
- Exploration bonuses when no food is visible
- Penalties for hazardous cells
- Distance-based incentives to return home when carrying food
- Energy-conservation incentives when resources are low

### Environment Interaction

The agent interacts with its environment through:

1. **Observation**: Perceives cell types via one-hot encoding
2. **Action**: Moves or rests based on planning outcomes
3. **Consequences**: Updates internal state based on:
   - Metabolic costs (energy decrease, hunger increase)
   - Rewards (food collection, consumption at home)
   - Hazard penalties (pain increase)

## Implementation Details

- **Grid Representation**: 25×25 grid with color-coded cell types (home, food, hazard, empty)
- **One-Hot Encoding**: Converts categorical cell types to vector representation
- **Streamlit Interface**: Real-time visualization and control of the simulation
- **Numpy-based Persistence**: Compressed storage of agent memory and state

## Usage

To run the simulation:

```bash
streamlit run main.py
```

Control options:
- **Start**: Begin the simulation
- **Stop**: Pause the agent
- **Reset**: Clear state and reinitialize the world

## Project Structure

```
.
├── main.py           # Main application file with agent and world logic
├── agent_state.npz   # Persisted agent state (created at runtime)
└── README.md         # This documentation file
```

## Future Work

Potential areas for enhancement:
- Multi-agent interactions
- Dynamic environment with resource regeneration
- More sophisticated memory structures
- Learning from repeated interactions
