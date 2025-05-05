# Hop to it: Autonomous Agents with Modern Hopfield Memory

This simulation demonstrates emergent behavior and communication patterns between multiple autonomous agents in a complex environment. Agents learn and adapt over time, developing their own strategies for survival and cooperation without hard-coded behaviors.

![Simulation Screenshot](simulation-example-main.png)

## Overview

This project implements a multi-agent simulation where intelligent agents navigate a terrain-based world, learn from their environment, communicate with each other, and develop emergent social behaviors. The agents use various AI techniques such as reinforcement learning, neural networks for communication, and Hopfield networks for memory.

## Features

- **Terrain-Based World**: Agents navigate a world with various terrain types (dirt, water, stone, etc.) each with unique properties like friction, temperature, and risk factors.
- **Adaptive Agent Behavior**: Agents learn from their environment using Q-learning and Hopfield networks, developing individual strategies for survival.
- **Goal-Based Planning System**: Agents set and pursue goals such as finding food, returning home, or resting.
- **Neural Communication**: Agents can communicate with each other by broadcasting signals that are learned and interpreted through neural networks.
- **Emergent Social Behavior**: Agents may develop emergent social patterns like food sharing and communication.
- **Dynamic Environment**: Weather systems and day/night cycles affect agent behavior and terrain properties. (TODO)
- **Interactive Visualization**: Real-time visualization of the simulation with detailed agent statistics, world information, and interactive controls.
- **Data Analysis Tools**: Integrated tools for analyzing agent states, learning patterns, and behavior trends.

## System Architecture

### World Model

The terrain system uses `TerrainCell` objects to represent each cell in the grid world, with properties including:
- Material type (dirt, stone, water, etc.)
- Physical properties (hardness, density, friction)
- Temperature and thermal conductivity
- Risk factors and passability

The world also models weather patterns and day/night cycles that affect both terrain and agent behavior.

### Agent Architecture

Each agent has:
- **Physical Properties**: Energy, hunger, pain, and inventory management
- **Perception Systems**: Agents observe their local environment to make decisions
- **Memory Systems**:
  - Episodic memories via Hopfield networks
  - Q-learning for reinforcement learning
  - Experience tracking for different terrain types
- **Planning System**: Goal-based planning with priorities and plan execution
- **Neural Communication**: Agents use encoder/decoder neural networks to broadcast and interpret signals

### Emergent Behavior

The simulation focuses on minimizing hard-coded behaviors, instead letting strategies and social patterns emerge from basic reinforcement learning and environmental pressures.

Some emergent behaviors you might observe:
- Food gathering and storage strategies
- Shelter-seeking during adverse weather
- Communication about food locations or dangers
- Social bonds and sharing behaviors
- Territory formation and defense

## Usage

### Running the Simulation

Run the simulation with:

```bash
python main_pygame.py
```

### Command Line Options

- `--load [FILENAME]`: Load a saved simulation state

### Controls

- **Space**: Pause/resume simulation
- **Right Arrow**: Step forward one tick when paused
- **Click on Agent**: Select agent to view detailed statistics
- **M**: Toggle minimap display

### User Interface

- **Play/Pause Button**: Toggle simulation running state
- **Step Button**: Advance simulation by one tick
- **Speed Slider**: Adjust simulation speed
- **Add Agent Button**: Add a new agent to the world
- **Remove Agent Button**: Remove the currently selected agent
- **Save/Load Buttons**: Save or load simulation state
- **Reset Button**: Reset the simulation to a fresh state
- **Toggle Minimap Button**: Show/hide the minimap overlay

### Agent State Analysis

You can use the agent state analyzer to examine the internal state of agents:

```bash
python agent_state_analysis.py                # looks for ./agent_state.npz
python agent_state_analysis.py path/to/file   # inspect different file
python agent_state_analysis.py --full         # dump all arrays verbatim
python agent_state_analysis.py --plot         # visualize memory patterns
python agent_state_analysis.py --export       # export data to CSV files
python agent_state_analysis.py --learning     # detailed learning analysis
```

### World Builder

Create and customize world environments:

```bash
python world_builder.py
```

## Extending the Simulation

You can customize various aspects of the simulation:
- Edit terrain generation in the `World` class
- Modify agent learning parameters in the `Agent` class
- Add new materials and terrain types
- Design new weather patterns and environmental effects
- Add new agent capabilities or communication mechanisms
- Extend the planning system with new goal types

## Project Structure

- `main.py`: Main simulation loop and visualization
- `module_agent.py`: Agent implementation with learning mechanisms
- `module_new_world.py`: Terrain-based world implementation
- `module_hopfield.py`: Memory system based on Hopfield networks
- `module_planner.py`: Goal-based planning system for agents
- `module_coms.py`: Neural communication system (requires PyTorch)
- `agent_state_analysis.py`: Tools for analyzing agent state
- `visualization_utils.py`: Helper functions for visualization
- `world_builder.py`: Interactive world creation tool
