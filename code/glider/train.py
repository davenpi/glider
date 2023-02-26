"""
This script runs the training of the agent and saves the agent history.
"""
import numpy as np
import matplotlib.pyplot as plt
from glider import Glider
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import pickle


glider = Glider(u0=0.25, v0=0.25, w0=0.1)
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./big_state_models/",
    name_prefix="rl_model",
)

model = PPO("MlpPolicy", glider, verbose=0, tensorboard_log="big_state_logs/")
model.learn(total_timesteps=6e5, callback=checkpoint_callback, progress_bar=True)

glider = Glider(u0=0.25, v0=-0.25, w0=0.1)
model = PPO.load("big_state_models/rl_model_600000_steps.zip", env=glider)
done = False
obs = glider.reset()
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _ = glider.step(action.item())


def save_history(glider: Glider, filename: str) -> None:
    """
    Save the full history of the glider in a dictionary
    """
    history = {
        "u": glider.u,
        "v": glider.v,
        "w": glider.w,
        "x": glider.x,
        "y": glider.y,
        "theta": glider.theta,
        "beta": glider.beta,
    }
    with open(filename + ".pkl", "wb") as f:
        pickle.dump(history, f)


save_history(glider=glider, filename="history")
