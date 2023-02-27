"""
This file creates movies showing glider trajectories over time.
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import argparse
from math import floor
from glider import Glider
from stable_baselines3 import PPO

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fname", help="Video file name.", type=str)
parser.add_argument("-n", "--model_n", help="Model to load", type=int)
args = parser.parse_args()
n = args.model_n
video_name = args.fname


glider = Glider(u0=0.25, v0=-0.25, w0=0.1)
model = PPO.load("big_state_models/rl_model_" + str(n) + "_steps.zip", env=glider)
done = False
obs = glider.reset()
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _ = glider.step(action.item())


x = np.array(glider.x)
x_min, x_max = x.min(), x.max()
y = np.array(glider.y)
y_min, y_max = y.min(), y.max()
theta = np.array(glider.theta)
beta = np.array(glider.beta)
n = beta.size
width = np.ones(shape=n)
height = beta * width

fig, ax = plt.subplots()
xdata, ydata = x, y
(ln,) = ax.plot([], [], "ro")
pad = 2


def init():
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min, y_max + pad)
    return (ln,)


def update(frame):
    ax.clear()
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad / 2, y_max + pad)
    ln.set_data(xdata[frame], ydata[frame])
    e = Ellipse(
        xy=(x[frame], y[frame]),
        width=width[frame],
        height=height[frame],
        angle=np.rad2deg(np.mod(theta[frame], 2 * np.pi)),
    )
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    ax.scatter(5, y_min)
    return (ln,)


ani = FuncAnimation(
    fig, update, frames=n, init_func=init, blit=True, interval=20, repeat=False
)
writervideo = animation.FFMpegWriter(fps=30)
ani.save(filename=video_name, writer=writervideo)
plt.close()


"""
Ellipse drawing code. This is for the sparse visualization
"""
ells = [
    Ellipse(
        xy=(x[i], y[i]),
        width=width[i],
        height=height[i],
        angle=theta[i],
    )
    for i in np.arange(start=0, stop=n, step=floor(n / 10))
]

fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})
for e in ells:
    ax.add_artist(e)

ax.set_xlim(x.min() - pad, x.max() + pad)
ax.set_ylim(y.min(), y.max() + pad)

plt.savefig("sparse_flutter_viz.png")
plt.close()


fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(12, 10))

fig.text(0.5, 0.04, "Time", ha="center")
fig.suptitle("Logged info from trajectory", fontsize=20)
ax[0, 0].plot(glider.t_hist, theta)
ax[0, 0].axhline(y=np.pi / 2)
ax[0, 0].axhline(y=-np.pi / 2)
ax[0, 0].set_ylabel(r"$\theta$")
ax[0, 1].plot(glider.t_hist, x)
ax[0, 1].set_ylabel("X")
ax[1, 0].plot(glider.t_hist, y)
ax[1, 0].set_ylabel("Y")
ax[1, 1].plot(glider.t_hist, glider.beta)
ax[1, 1].set_ylabel(r"$\beta$")
ax[2, 0].plot(glider.t_hist, glider.v)
ax[2, 0].set_ylabel("V")
ax[2, 1].plot(glider.t_hist, glider.u)
ax[2, 1].set_ylabel("U")

plt.show()


# def save_history(glider: Glider, filename: str) -> None:
#     """
#     Save the full history of the glider in a dictionary
#     """
#     history = {
#         "u": glider.u,
#         "v": glider.v,
#         "w": glider.w,
#         "x": glider.x,
#         "y": glider.y,
#         "theta": glider.theta,
#         "beta": glider.beta,
#     }
#     with open(filename + ".pkl", "wb") as f:
#         pickle.dump(history, f)


# save_history(glider=glider, filename="history")
