import vizdoom as vzd

print("Initializing DOOM for Human Play...")

game = vzd.DoomGame()

# Load the basic scenario
game.load_config(vzd.scenarios_path + "/basic.cfg")

# Set the mode to SPECTATOR
# This mode allows the human to play using the keyboard/mouse 
# while the Python script just watches and keeps the engine running.
game.set_mode(vzd.Mode.SPECTATOR)

game.init()

print("============================================")
print("DOOM is ready! Click on the game window.")
print("Controls:")
print(" - Arrow Keys to move/turn")
print(" - CTRL to Shoot")
print("============================================")

episodes = 5
for i in range(episodes):
    print(f"Starting Episode {i+1}...")
    game.new_episode()
    
    while not game.is_episode_finished():
        # advance_action() tells the game to move one frame forward,
        # and lets the human's keyboard/mouse inputs take control!
        game.advance_action()
        
    print(f"Episode {i+1} finished! Score: {game.get_total_reward()}")

print("Closing DOOM.")
game.close()
