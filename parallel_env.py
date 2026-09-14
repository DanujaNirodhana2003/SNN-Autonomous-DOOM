import concurrent.futures
from environment import DoomEnvironment
import torch

class VectorizedDoomEnv:
    def __init__(self, num_envs=4, config_file="basic.cfg"):
        """
        Creates multiple DOOM environments that run in parallel.
        :param num_envs: Number of games to run at the same time.
        """
        self.num_envs = num_envs
        # Create independent environments. Render must be False for parallel!
        self.envs = [DoomEnvironment(config_file=config_file, render=False) for _ in range(num_envs)]
        
        # ThreadPoolExecutor hides the latency of the C++ game engine
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_envs)
        
    def reset(self):
        """Resets all environments in parallel and returns a batch of states."""
        futures = [self.executor.submit(env.reset) for env in self.envs]
        states = [f.result() for f in futures]
        
        # Stack into a single batch tensor: shape (num_envs, 1, 64, 64)
        return torch.stack(states, dim=0)
        
    def step(self, actions):
        """
        Takes a list of actions and applies them to all environments at once.
        Auto-resets any environment that finishes an episode.
        Uses the reward-shaped step() from DoomEnvironment.
        """
        def _step_single(env, action):
            # Use the environment's step() which includes reward shaping!
            next_state, shaped_reward, done = env.step(action)
            
            # If the episode finished, automatically restart it
            if done:
                next_state = env.reset()
                
            return next_state, shaped_reward, done

        # Run all steps in parallel threads
        futures = [self.executor.submit(_step_single, env, action) for env, action in zip(self.envs, actions)]
        results = [f.result() for f in futures]
        
        # Unzip results
        next_states, rewards, dones = zip(*results)
        
        # Convert to PyTorch tensors for the Neural Network
        next_states_tensor = torch.stack(next_states, dim=0)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)
        
        return next_states_tensor, rewards_tensor, dones_tensor

    def close(self):
        """Shuts down all games."""
        for env in self.envs:
            env.close()
        self.executor.shutdown()
