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
        
        # --- Reward Shaping: Add extra game variables we need to track ---
        # These might already be in the .cfg, but adding them programmatically
        # ensures it works everywhere (Kaggle, Server, Laptop)
        self.game.add_available_game_variable(vzd.GameVariable.HITCOUNT)
        self.game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
        self.game.add_available_game_variable(vzd.GameVariable.DAMAGECOUNT)
        
        # Stronger death penalty for reward shaping
        self.game.set_death_penalty(5.0)
        
        self.game.init()
        
        # In basic.cfg, we only need 3 actions: Turn Left, Turn Right, Shoot
        self.actions = [
            [1, 0, 0], # Action 0: Turn Left
            [0, 1, 0], # Action 1: Turn Right
            [0, 0, 1]  # Action 2: Shoot
        ]
        
        # Resize to 64x64 for the SNN to keep the network small and fast
        self.resize = T.Resize((64, 64), antialias=True)
        
        # --- Reward Shaping: Track game variables ---
        self.prev_hitcount = 0
        self.prev_ammo = 0
        self.prev_killcount = 0

    def reset(self):
        """Starts a new episode and returns the first frame."""
        self.game.new_episode()
        
        # Reset reward shaping trackers using named variable access
        self.prev_ammo = self.game.get_game_variable(vzd.GameVariable.AMMO2)
        self.prev_hitcount = self.game.get_game_variable(vzd.GameVariable.HITCOUNT)
        self.prev_killcount = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        
        return self.get_state()

    def step(self, action_idx):
        """
        Executes an action in the game with REWARD SHAPING.
        :param action_idx: Integer 0, 1, or 2
        :return: (next_state, shaped_reward, is_done)
        """
        # Execute the action and get the base reward from the game engine
        base_reward = self.game.make_action(self.actions[action_idx])
        done = self.game.is_episode_finished()
        
        # Start with the base reward from the game (kill/death rewards from .cfg)
        shaped_reward = base_reward
        
        if not done:
            # Read current game variables by NAME (no index issues!)
            current_ammo = self.game.get_game_variable(vzd.GameVariable.AMMO2)
            current_hitcount = self.game.get_game_variable(vzd.GameVariable.HITCOUNT)
            current_killcount = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
            
            # --- Reward Shaping Signals ---
            
            # 1. HIT REWARD: Bullet hit an enemy! (+2.0 per hit)
            hits = current_hitcount - self.prev_hitcount
            if hits > 0:
                shaped_reward += 2.0 * hits
            
            # 2. MISS PENALTY: Ammo was used but didn't hit anything (-0.1)
            ammo_used = self.prev_ammo - current_ammo
            if ammo_used > 0 and hits == 0:
                shaped_reward -= 0.1 * ammo_used
            
            # 3. KILL BONUS: Enemy was killed! (+5.0 per kill)
            kills = current_killcount - self.prev_killcount
            if kills > 0:
                shaped_reward += 5.0 * kills
            
            # 4. SURVIVAL REWARD: Still alive = keep scanning! (+0.01)
            shaped_reward += 0.01
            
            # Update trackers for the next step
            self.prev_ammo = current_ammo
            self.prev_hitcount = current_hitcount
            self.prev_killcount = current_killcount
            
            next_state = self.get_state()
        else:
            # If the episode is over, return a blank screen
            next_state = torch.zeros((1, 64, 64), dtype=torch.float32)
            
        return next_state, shaped_reward, done

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
