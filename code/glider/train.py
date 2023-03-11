"""
This script runs the training of the agent and saves the agent history.
"""
from glider import Glider
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-s", "--steps", default=5e5, type=float, help="Number of learning steps"
)

args = parser.parse_args()
n = int(args.steps)

glider = Glider()
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./energy_reward_models/",
    name_prefix="rl_model",
)

# model = PPO.load(
#     "nice_result/rl_model_600000_steps.zip",
#     env=glider,
#     tensorboard_log="big_state_logs/",
# )
model = PPO("MlpPolicy", env=glider, tensorboard_log="energy_reward_logs/")
model.learn(total_timesteps=n, callback=checkpoint_callback, progress_bar=True)
