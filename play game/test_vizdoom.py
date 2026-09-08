import vizdoom as vzd
from time import sleep
import random

print("Initializing DOOM...")

# 1. Create a game instance
game = vzd.DoomGame()

# 2. Load the default basic configuration
# This comes bundled with the pip install!
game.load_config(vzd.scenarios_path + "/basic.cfg")

# 3. Initialize the game engine
game.init()

print("DOOM started! Running some random actions...")

# 4. Play the game for a few frames
# The basic scenario has 3 actions: Turn Left, Turn Right, Shoot
actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

for step in range(100):
    # Pick a random action
    action = random.choice(actions)
    
    # Send the action to the game and get the reward
    reward = game.make_action(action)
    
    # Sleep a bit so you can actually see it on the screen (otherwise it runs too fast!)
    sleep(0.05)

print("Finished! Closing DOOM.")
game.close()
