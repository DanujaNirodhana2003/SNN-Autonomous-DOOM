import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import time
from collections import deque
from parallel_env import VectorizedDoomEnv
from model import SpikingQNetwork

# --- Parallel Hyperparameters ---
NUM_ENVS = 4           # Run 4 games at the exact same time!
BATCH_SIZE = 64
GAMMA = 0.99           
EPSILON_START = 1.0    
EPSILON_END = 0.1      
EPSILON_DECAY = 0.998  
LR = 0.001             
MEMORY_SIZE = 20000    
TOTAL_EPISODES = 500   

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push_batch(self, states, actions, rewards, next_states, dones):
        # Break the batch down into single experiences to store in memory
        for i in range(len(states)):
            self.buffer.append((states[i:i+1], actions[i], rewards[i], next_states[i:i+1], dones[i]))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, n_s, d = zip(*batch)
        return (
            torch.cat(s, dim=0),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.cat(n_s, dim=0),
            torch.tensor(d, dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

def train_parallel():
    print(f"Initializing {NUM_ENVS} parallel environments...")
    # NOTE: Render must be False for parallel environments!
    env = VectorizedDoomEnv(num_envs=NUM_ENVS, config_file="basic.cfg")
    
    model = SpikingQNetwork()
    target_model = SpikingQNetwork()
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    memory = ReplayBuffer(MEMORY_SIZE)
    
    epsilon = EPSILON_START
    states = env.reset() # Shape: (4, 1, 64, 64)
    
    print("======================================")
    print("Starting PARALLEL SNN Training on DOOM...")
    print("======================================")
    
    start_time = time.time()
    total_rewards = [0.0] * NUM_ENVS
    episodes_completed = 0
    
    # Create a progress bar to see real-time updates!
    from tqdm import tqdm
    pbar = tqdm(total=TOTAL_EPISODES, desc="Training Progress")
    
    # We train until we hit 500 total episodes across all parallel games
    while episodes_completed < TOTAL_EPISODES:
        # 1. Epsilon-Greedy Batch Actions
        if random.random() < epsilon:
            # Random action for each environment
            actions = [random.randint(0, 2) for _ in range(NUM_ENVS)]
        else:
            with torch.no_grad():
                # Ask SNN to predict actions for ALL environments at once!
                _, mem = model(states, num_steps=10)
                q_values = mem.mean(dim=0) 
                actions = torch.argmax(q_values, dim=1).tolist()
                
        # 2. Step all environments in parallel
        next_states, rewards, dones = env.step(actions)
        
        # Track rewards and count completed episodes
        for i in range(NUM_ENVS):
            total_rewards[i] += rewards[i].item()
            if dones[i].item() == 1.0:
                episodes_completed += 1
                
                # Update the progress bar and show the latest reward/epsilon
                pbar.update(1)
                pbar.set_postfix({'Reward': f"{total_rewards[i]:.1f}", 'Eps': f"{epsilon:.2f}"})
                
                total_rewards[i] = 0.0 # Reset tracking for this env
                
        # 3. Store batch in memory
        memory.push_batch(states, actions, rewards, next_states, dones)
        states = next_states
        
        # 4. Train the SNN
        if len(memory) >= BATCH_SIZE:
            s_batch, a_batch, r_batch, ns_batch, d_batch = memory.sample(BATCH_SIZE)
            
            _, mem_current = model(s_batch, num_steps=10)
            q_values = mem_current.mean(dim=0)
            q_val = q_values.gather(1, a_batch.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                _, mem_next = target_model(ns_batch, num_steps=10)
                q_next = mem_next.mean(dim=0)
                max_q_next = q_next.max(1)[0]
                target_q = r_batch + (GAMMA * max_q_next * (1 - d_batch))
            
            loss = criterion(q_val, target_q)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Decay Epsilon gradually per step
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
            
        # Sync Target Network
        if episodes_completed > 0 and episodes_completed % 10 == 0:
            target_model.load_state_dict(model.state_dict())
            
    pbar.close()        
    print("Parallel Training Complete! Saving model to snn_parallel_model.pth...")
    torch.save(model.state_dict(), "snn_parallel_model.pth")
    env.close()

if __name__ == "__main__":
    train_parallel()
