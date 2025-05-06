# Hop to it: Modern Hopfield Auto‑Agent

A minimal, extensible multi‑agent simulation framework leveraging Modern Hopfield‑inspired memory and vector‑quantization communication.

## Overview

This project is a closed‑loop system where agents perceive a world, store observations in a Modern Hopfield‑style memory, and broadcast compressed symbols when their surprise or reward thresholds are exceeded. Key components:

* **Agent** (`module_agent.py`): Maintains a trace of observations, detects surprise, packages full sequences, and communicates via a VQ codebook.
* **Communication** (`module_comm.py`): Defines a vector‑quantization encoder/decoder that assigns observations sequences to discrete symbols.
* **Memory** (`module_hopfield.py`): Implements a Modern Hopfield network for associative memory storage and retrieval.
* **World** (`module_world.py`): Simulates an environment that generates observations, rewards, and time steps for agents.
* **Entry Point** (`main.py`): Configures the world and agents, runs the simulation loop, and logs interactions.

## Features

* Surprise‑driven communication: agents only broadcast when encountering novel or unexpected inputs.
* Discrete symbol encoding with k‑means codebook for scalable message passing.
* Modern Hopfield network for robust memory recall of past observation patterns.
* Pluggable world models—easily extendable to custom environments.

## Usage

```bash
streamlit run main.py
```

## Project Structure

```
├── main.py            # Simulation entry point
├── module_agent.py    # Agent class and communication logic
├── module_comm.py     # VQ encoder/decoder implementation
├── module_hopfield.py # Modern Hopfield network memory
├── module_world.py    # World/environment simulator
└── requirements.txt   # Python dependencies
```

## Configuration

Adjust thresholds and hyperparameters at the top of `module_agent.py` and `module_comm.py`:

**Examples:**
* `SURPRISE_THRESHOLD`, `REWARD_THRESHOLD`
* `CODEBOOK_SIZE`, `SEQ_LEN`, `EMBED_DIM`
