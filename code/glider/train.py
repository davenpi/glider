"""
This script runs the training of the agent and saves the agent history.
"""
import numpy as np
import matplotlib.pyplot as plt
from glider import Glider
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import pickle
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-s", "--steps", default=5e5, type=float, help="Number of learning steps"
)

args = parser.parse_args()
n = int(args.steps)

glider = Glider(u0=0.25, v0=0.25, w0=0.1)
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./big_state_models/",
    name_prefix="rl_model",
)

# model = PPO("MlpPolicy", glider, verbose=0, tensorboard_log="big_state_logs/")
model = PPO.load("pretrained_models/rl_model_900000_steps.zip", env = glider, tensorboard_log = "big_state_logs/")
model.learn(total_timesteps=n, callback=checkpoint_callback, progress_bar=True)
