import gymnasium as gym
import gymsnake # Register the Snake environment
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# creates an environment and wraps it with gymnasium.wrappers.time_limit.TimeLimit
 
# if you want multiplayer on 7x7 grid
env = gym.make('snake-v1', n_snakes=1, grid_size=(7, 7), disable_env_checker=True, body_start_length=1)

# Define the PPO model
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    ent_coef=0.05,  # Entropy coefficient for exploration
    learning_rate=0.0001,  # Adjust learning rate if necessary
    n_steps=2048,  # Number of steps to run for each environment per update
    batch_size=64,  # Batch size for optimization
    gamma=0.99,  # Discount factor
    gae_lambda=0.95,  # Lambda for Generalized Advantage Estimation
    clip_range=0.2,  # Clip range for PPO
    n_epochs=10,  # Number of optimization epochs
)

# Train the model
model.learn(total_timesteps=1000000)

# Save the trained model
model.save("snake_dqn")

# Load and test the trained model
model = PPO.load("snake_dqn")

# Reset environment and run inference with the trained agent
obs, _ = env.reset()
terminated = False

while not terminated:
    # Use the model to predict the next action
    action, _states = model.predict(obs, deterministic=False)

    # Perform the action in the environment and unpack the values
    obs, alive_snakes, rewards, terminated, info = env.step(action)
    
    env.render()  # Render the environment
    
    # Check if the game has ended or if snakes are alive
    if alive_snakes < 0:  # Check if any snakes are alive
        print("Game Over! Resetting the environment.")
        obs, _ = env.reset()  # Reset the environment for a new game
        terminated = False  # Reset terminated for the new game loop

    # Optional: If you want to handle the terminated state separately
    if terminated:
        print("Game terminated! Resetting the environment.")
        obs, _ = env.reset()  # Reset the environment for a new game
        terminated = False  # Reset terminated for the new game loop