import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import random

MAX_E, MAX_H, MAX_P = 100, 100, 100
GRID = 25

# ───────────────────── Neural Communication System ─────────────────────
class SignalEncoder(nn.Module):
    """Neural network to encode agent state into signal vectors"""
    
    def __init__(self, state_dim=10, hidden_dim=20, signal_dim=5):
        super(SignalEncoder, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, signal_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))  # Signal values between -1 and 1
        return x

class SignalDecoder(nn.Module):
    """Neural network to decode received signals into meaning"""
    
    def __init__(self, signal_dim=5, hidden_dim=20, output_dim=3):
        super(SignalDecoder, self).__init__()
        self.fc1 = nn.Linear(signal_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Raw logits, will apply softmax later
        return x

class NeuralCommunicationSystem:
    """System for emergent communication between agents using neural networks"""
    
    def __init__(self, state_dim=10, signal_dim=5, output_dim=3):
        self.state_dim = state_dim
        self.signal_dim = signal_dim
        self.output_dim = output_dim  # Number of meaning categories
        
        # Agents' neural networks
        self.encoders = {}  # agent_id -> SignalEncoder
        self.decoders = {}  # agent_id -> SignalDecoder
        self.optimizers_enc = {}  # agent_id -> encoder optimizer
        self.optimizers_dec = {}  # agent_id -> decoder optimizer
        
        # Signal memories for each agent
        self.signal_memory = {}  # agent_id -> list of (signal, context, outcome) tuples
        
        # Active signals in the environment
        self.active_signals = []  # List of {sender, signal, position, time, range}
        
        # Meaning categories
        self.meaning_categories = ["food", "danger", "help"]
    
    def initialize_agent(self, agent_id):
        """Set up neural networks for a new agent"""
        # Create neural networks
        self.encoders[agent_id] = SignalEncoder(self.state_dim, 20, self.signal_dim)
        self.decoders[agent_id] = SignalDecoder(self.signal_dim, 20, self.output_dim)
        
        # Create optimizers
        self.optimizers_enc[agent_id] = optim.Adam(self.encoders[agent_id].parameters(), lr=0.001)
        self.optimizers_dec[agent_id] = optim.Adam(self.decoders[agent_id].parameters(), lr=0.001)
        
        # Initialize memory
        self.signal_memory[agent_id] = []
    
    def agent_state_to_vector(self, agent):
        """Convert agent state to input vector for the encoder"""
        # Extract key state information
        energy_pct = agent.energy / MAX_E
        hunger_pct = agent.hunger / MAX_H
        pain_pct = agent.pain / MAX_P
        carrying_food = 1.0 if agent.carrying else 0.0
        stored_food = min(agent.store / 10, 1.0)  # Normalize food store
        
        # Distance to home
        home_x, home_y = agent.w.home
        home_dist = (abs(agent.pos[0] - home_x) + abs(agent.pos[1] - home_y)) / (GRID * 2)
        
        # Distance to nearest food
        food_dist = agent.w.nearest_food_distance(tuple(agent.pos)) / GRID
        
        # Create state vector
        state = torch.tensor([
            energy_pct, 
            hunger_pct, 
            pain_pct, 
            carrying_food, 
            stored_food, 
            home_dist, 
            food_dist,
            agent.pos[0] / GRID,  # Normalized position
            agent.pos[1] / GRID,
            agent.tick_count / 1000.0  # Time factor
        ], dtype=torch.float32)
        
        return state
    
    def generate_signal(self, agent):
        """Generate a signal based on agent's current state"""
        # Get agent state as input vector
        state_vector = self.agent_state_to_vector(agent)
        
        # Use encoder network to generate signal
        with torch.no_grad():  # Don't track gradients for inference
            encoder = self.encoders[agent.id]
            signal = encoder(state_vector)
        
        return signal.numpy()
    
    def broadcast_signal(self, agent_id, signal, position, range=3):
        """Place a signal in the environment for other agents to perceive"""
        # Add signal to active signals list
        self.active_signals.append({
            "sender": agent_id,
            "signal": signal,
            "position": position,
            "time": time.time(),
            "range": range
        })
    
    def perceive_signals(self, agent):
        """Get all signals that an agent can perceive"""
        received = []
        
        for signal_data in self.active_signals:
            # Skip if agent is the sender
            if signal_data["sender"] == agent.id:
                continue
                
            # Check if signal is in range
            sender_pos = signal_data["position"]
            agent_pos = agent.pos
            distance = abs(sender_pos[0] - agent_pos[0]) + abs(sender_pos[1] - agent_pos[1])
            
            if distance <= signal_data["range"]:
                received.append({
                    "signal": signal_data["signal"],
                    "sender": signal_data["sender"],
                    "position": sender_pos,
                    "distance": distance
                })
        
        return received
    
    def interpret_signal(self, agent, signal_data):
        """Interpret a received signal using decoder network"""
        signal = signal_data["signal"]
        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        
        # Use decoder to interpret signal
        with torch.no_grad():  # Don't track gradients
            decoder = self.decoders[agent.id]
            logits = decoder(signal_tensor)
            probs = F.softmax(logits, dim=0)
        
        # Get meaning category with highest probability
        meaning_idx = torch.argmax(probs).item()
        meaning = self.meaning_categories[meaning_idx]
        confidence = probs[meaning_idx].item()
        
        return {
            "meaning": meaning,
            "confidence": confidence,
            "probabilities": probs.numpy(),
            "sender": signal_data["sender"],
            "position": signal_data["position"]
        }
    
    def record_experience(self, agent_id, signal, context, outcome_reward):
        """Record signal experience for learning"""
        # Convert to tensors if not already
        if not isinstance(signal, torch.Tensor):
            signal = torch.tensor(signal, dtype=torch.float32)
        
        if not isinstance(context, torch.Tensor):
            context = torch.tensor(context, dtype=torch.float32)
        
        # Store experience
        self.signal_memory[agent_id].append({
            "signal": signal,
            "context": context,
            "reward": outcome_reward,
            "time": time.time()
        })
        
        # Limit memory size
        if len(self.signal_memory[agent_id]) > 100:
            self.signal_memory[agent_id] = self.signal_memory[agent_id][-100:]
    
    def train_encoder(self, agent_id, batch_size=8):
        """Train encoder network based on signal outcomes"""
        memory = self.signal_memory[agent_id]
        
        # Need enough samples to train
        if len(memory) < batch_size:
            return
        
        # Sample batch of experiences
        batch_indices = np.random.choice(len(memory), batch_size, replace=False)
        batch = [memory[i] for i in batch_indices]
        
        # Prepare inputs and targets
        contexts = torch.stack([b["context"] for b in batch])
        signals = torch.stack([b["signal"] for b in batch])
        rewards = torch.tensor([b["reward"] for b in batch], dtype=torch.float32)
        
        # Zero gradients
        self.optimizers_enc[agent_id].zero_grad()
        
        # Forward pass
        encoder = self.encoders[agent_id]
        decoder = self.decoders[agent_id]
        
        # Generate signals from contexts
        predicted_signals = encoder(contexts)
        
        # Calculate loss based on reward prediction
        # Signals that led to good outcomes should be reinforced
        loss = 0
        for i in range(batch_size):
            # Predict meaning from signal
            signal_meaning = decoder(signals[i])
            
            # Calculate reward prediction loss
            # Positive reward should reinforce this signal-context pair
            if rewards[i] > 0:
                loss += F.mse_loss(predicted_signals[i], signals[i])
            else:
                # Negative reward should discourage this signal-context pair
                # by moving away from it
                loss += -F.mse_loss(predicted_signals[i], signals[i])
        
        # Backward pass and optimize
        loss.backward()
        self.optimizers_enc[agent_id].step()
    
    def train_decoder(self, agent_id, batch_size=8):
        """Train decoder network to predict meaning/value of signals"""
        memory = self.signal_memory[agent_id]
        
        # Need enough samples to train
        if len(memory) < batch_size:
            return
        
        # Sample batch of experiences
        batch_indices = np.random.choice(len(memory), batch_size, replace=False)
        batch = [memory[i] for i in batch_indices]
        
        # Prepare inputs and targets
        signals = torch.stack([b["signal"] for b in batch])
        rewards = torch.tensor([b["reward"] for b in batch], dtype=torch.float32)
        
        # Create target values based on reward
        # We'll use a simple classification:
        # reward > 0 -> category 0 (food/good)
        # reward < -5 -> category 1 (danger/bad)
        # otherwise -> category 2 (neutral/help)
        targets = torch.zeros(batch_size, dtype=torch.long)
        for i, reward in enumerate(rewards):
            if reward > 5:
                targets[i] = 0  # Food/beneficial
            elif reward < -5:
                targets[i] = 1  # Danger/harmful
            else:
                targets[i] = 2  # Help/neutral
        
        # Zero gradients
        self.optimizers_dec[agent_id].zero_grad()
        
        # Forward pass
        decoder = self.decoders[agent_id]
        logits = decoder(signals)
        
        # Calculate loss
        loss = F.cross_entropy(logits, targets)
        
        # Backward pass and optimize
        loss.backward()
        self.optimizers_dec[agent_id].step()
    
    def learn_from_observation(self, observer_id, observed_id):
        """Learn from observing another agent's signals and outcomes"""
        # This is only effective if the observed agent has successful experiences
        observed_memory = self.signal_memory[observed_id]
        
        # Nothing to learn
        if not observed_memory:
            return
        
        # Filter for successful experiences (positive rewards)
        successful_exp = [exp for exp in observed_memory if exp["reward"] > 2.0]
        
        # Nothing successful to learn
        if not successful_exp:
            return
        
        # Sample successful experience to imitate
        exp = random.choice(successful_exp)
        
        # Add to observer's memory to learn from
        self.record_experience(
            observer_id, 
            exp["signal"], 
            exp["context"], 
            exp["reward"] * 0.8  # Slightly lower reward for observed vs. experienced
        )
    
    def clean_old_signals(self, max_age=10.0):
        """Remove signals that are too old"""
        current_time = time.time()
        self.active_signals = [s for s in self.active_signals 
                              if current_time - s["time"] < max_age]
