import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from environment import DoomEnvironment
from model import SpikingQNetwork

# --- Hyperparameters ---
BATCH_SIZE = 32
GAMMA = 0.99           # Discount factor for future rewards
EPSILON_START = 1.0    # 100% random actions at start (Exploration)
EPSILON_END = 0.1      # Minimum 10% random actions
EPSILON_DECAY = 0.995  # Decay rate for epsilon
LR = 0.001             # Learning rate
MEMORY_SIZE = 10000    # Replay buffer capacity
NUM_EPISODES = 500     # Total episodes to train

class ReplayBuffer:
    """Stores past experiences so the network can learn from them."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

def train():
    # User requested to see the window during training (render=True).
    # Note: This will make the training process significantly slower!
    env = DoomEnvironment(config_file="basic.cfg", render=True) 
    
    model = SpikingQNetwork()
    target_model = SpikingQNetwork()
    target_model.load_state_dict(model.state_dict()) # Clone model to target
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    memory = ReplayBuffer(MEMORY_SIZE)
    
    epsilon = EPSILON_START
    
    print("======================================")
    print("Starting SNN Training on DOOM...")
    print("======================================")
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 1. Choose an action (Epsilon-Greedy)
            if random.random() < epsilon:
                action = random.randint(0, 2) # Random action
            else:
                with torch.no_grad():
                    # Ask the SNN what to do!
                    # We use the average membrane potential as our "Q-Value"
                    _, mem = model(state.unsqueeze(0), num_steps=10)
                    q_values = mem.mean(dim=0) 
                    action = torch.argmax(q_values).item()
                    
            # 2. Take the action in the game
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # 3. Remember the result
            memory.push(state, action, reward, next_state, done)
            state = next_state
            
            # 4. Train the SNN
            if len(memory) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                
                # What did the network predict?
                _, mem_current = model(states, num_steps=10)
                q_values = mem_current.mean(dim=0)
                q_val = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # What was the ACTUAL best future reward? (Target)
                with torch.no_grad():
                    _, mem_next = target_model(next_states, num_steps=10)
                    q_next = mem_next.mean(dim=0)
                    max_q_next = q_next.max(1)[0]
                    target_q = rewards + (GAMMA * max_q_next * (1 - dones))
                
                # Calculate Error (Loss) and Update Weights
                loss = criterion(q_val, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        # End of Episode: Decay randomness
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        # Sync Target Network every 10 episodes
        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())
            
        print(f"Episode {episode+1}/{NUM_EPISODES} | Reward: {total_reward:5.1f} | Epsilon: {epsilon:.2f}")

    print("Training Complete! Saving model to snn_doom_model.pth...")
    torch.save(model.state_dict(), "snn_doom_model.pth")
    env.close()

if __name__ == "__main__":
    train()
