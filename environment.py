import vizdoom as vzd
import numpy as np
import torch
import torchvision.transforms as T

class DoomEnvironment:
    def __init__(self, config_file="basic.cfg", render=True):
        """
        Wrapper for VizDoom tailored for Spiking Neural Networks.
        :param config_file: The scenario file to load (default: basic.cfg)
        :param render: If True, the game window will be visible (good for testing, slow for training)
        """
        self.game = vzd.DoomGame()
        self.game.load_config(vzd.scenarios_path + "/" + config_file)
        
        # User requested to see the window!
        self.game.set_window_visible(render)
        
        # Set screen to Grayscale to save memory and processing power
        self.game.set_screen_format(vzd.ScreenFormat.GRAY8) 
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        
        self.game.init()
        
        # In basic.cfg, we only need 3 actions: Turn Left, Turn Right, Shoot
        self.actions = [
            [1, 0, 0], # Action 0: Turn Left
            [0, 1, 0], # Action 1: Turn Right
            [0, 0, 1]  # Action 2: Shoot
        ]
        
        # Resize to 64x64 for the SNN to keep the network small and fast
        self.resize = T.Resize((64, 64), antialias=True)

    def reset(self):
        """Starts a new episode and returns the first frame."""
        self.game.new_episode()
        return self.get_state()

    def step(self, action_idx):
        """
        Executes an action in the game.
        :param action_idx: Integer 0, 1, or 2
        :return: (next_state, reward, is_done)
        """
        reward = self.game.make_action(self.actions[action_idx])
        done = self.game.is_episode_finished()
        
        if done:
            # If the episode is over, return a blank screen
            next_state = torch.zeros((1, 64, 64), dtype=torch.float32)
        else:
            next_state = self.get_state()
            
        return next_state, reward, done

    def get_state(self):
        """Grabs the current screen, resizes, and normalizes it."""
        state = self.game.get_state()
        
        # The screen buffer is (120, 160) because we set GRAY8
        img = state.screen_buffer 
        
        # Convert to PyTorch tensor and add a channel dimension: (1, 120, 160)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        
        # Resize to (1, 64, 64) and normalize pixels to be between 0.0 and 1.0
        img_resized = self.resize(img_tensor) / 255.0
        
        return img_resized

    def close(self):
        """Safely shuts down the game engine."""
        self.game.close()

if __name__ == "__main__":
    # Test the environment
    print("Testing DoomEnvironment...")
    env = DoomEnvironment(render=True)
    state = env.reset()
    
    print(f"Initial State Shape: {state.shape}") # Should be (1, 64, 64)
    print("Executing a random action...")
    
    next_state, reward, done = env.step(1) # Turn Right
    print(f"Reward Received: {reward}")
    
    env.close()
    print("Environment test passed successfully!")
