import vizdoom as vzd
import time

print("Loading DOOM for Human Play...")

game = vzd.DoomGame()

# Load a fun scenario with enemies coming from all sides
game.load_config(vzd.scenarios_path + "/defend_the_center.cfg")

# SPECTATOR mode: human plays, Python watches
game.set_mode(vzd.Mode.SPECTATOR)

# Make sure the window is visible with HUD
game.set_window_visible(True)
game.set_render_hud(True)

# Manually bind WASD keys and other controls so keyboard actually works!
game.add_game_args("+bind w +forward")
game.add_game_args("+bind s +back")
game.add_game_args("+bind a +moveleft")
game.add_game_args("+bind d +moveright")

# Long timeout so you can play freely (5 minutes)
game.set_episode_timeout(10500)

game.init()

print("============================================")
print("  DOOM - DEFEND THE CENTER")
print("============================================")
print("  Click on the DOOM window first!")
print("")
print("  Arrow Keys = Turn Left/Right")
print("  W/S        = Forward/Backward")  
print("  A/D        = Strafe Left/Right")
print("  CTRL       = Shoot")
print("  SPACE      = Use/Open")
print("  ESC        = Quit")
print("============================================")

episodes = 10
for i in range(episodes):
    print(f"\n--- Episode {i+1} / {episodes} ---")
    game.new_episode()
    
    while not game.is_episode_finished():
        # This is the KEY line!
        # advance_action() tells the engine to:
        # 1. Read the human's keyboard/mouse input
        # 2. Move the game forward by one frame
        # Without this, the game freezes!
        game.advance_action()
        
        state = game.get_state()
        if state is not None:
            reward = game.get_last_reward()
    
    print(f"Episode {i+1} finished! Score: {game.get_total_reward()}")

print("\nAll episodes done! Closing DOOM.")
game.close()
