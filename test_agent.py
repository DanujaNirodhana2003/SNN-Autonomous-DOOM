import torch
import time
from environment import DoomEnvironment
from model import SpikingQNetwork

def test_trained_agent():
    print("Loading the Trained SNN Brain...")
    
    # 1. Load the empty network architecture
    model = SpikingQNetwork()
    
    # 2. Load the trained memory/weights from the file!
    try:
        model.load_state_dict(torch.load("snn_parallel_model.pth", weights_only=True))
        print("Successfully loaded 'snn_parallel_model.pth'!")
    except FileNotFoundError:
        print("ERROR: Could not find 'snn_parallel_model.pth'. Please run 'parallel_train.py' first.")
        return
        
    model.eval() # Tell the network we are just testing, not training
    
    # 3. Create the environment WITH the Window open so we can watch!
    print("Starting VizDoom...")
    env = DoomEnvironment(config_file="basic.cfg", render=True)
    
    # 4. Let the AI play 5 games
    for episode in range(5):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            with torch.no_grad():
                # Ask the SNN to make a decision
                _, mem = model(state.unsqueeze(0), num_steps=10)
                q_values = mem.mean(dim=0)
                action = torch.argmax(q_values).item()
                
            # Execute the AI's chosen action
            state, reward, done = env.step(action)
            total_reward += reward
            
            # Slow down the game slightly so it's not too fast for human eyes
            time.sleep(0.05) 
            
        print(f"Test Game {episode+1} Finished | Score: {total_reward}")
        time.sleep(1) # Pause for 1 second before the next game starts
        
    env.close()

if __name__ == "__main__":
    test_trained_agent()
