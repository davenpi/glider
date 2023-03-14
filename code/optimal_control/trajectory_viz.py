"""
This file creates movies showing glider trajectories over time.
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import pickle
from math import floor

x_opt = np.load("optimal_sol.npy")
x = np.array(x_opt[3])
x_min, x_max = x.min(), x.max()
y = np.array(x_opt[4])
y_min, y_max = y.min(), y.max()
theta = np.array(x_opt[5])
beta = np.array(x_opt[6])
n = beta.size
width = np.sqrt(1 / beta)  # np.ones(shape=n)
height = np.sqrt(beta)  # beta * width

fig, ax = plt.subplots()
xdata, ydata = x, y
(ln,) = ax.plot([], [], "ro")
pad = 2


def init():
    ax.set_xlim(-x_min - pad, x_max + pad)
    ax.set_ylim(y_min, y_max + pad)
    return (ln,)


def update(frame):
    ax.clear()
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad / 2, y_max + pad)
    ln.set_data(xdata[frame], ydata[frame])
    e = Ellipse(
        xy=(x[frame], y[frame]),
        width=5 * width[frame],
        height=5 * height[frame],
        angle=np.rad2deg(np.mod(theta[frame], 2 * np.pi)),
    )

    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    ax.scatter(x[frame], y[frame], marker="o", c="green", s=2)
    # ax.scatter(glider.target_x, glider.terminal_y, marker="X", c="red", s=8)
    scale = 0.1
    ax.arrow(
        x=x[frame],
        y=y[frame],
        dx=2 * np.cos(theta[frame]),
        dy=2 * np.sin(theta[frame]),
        width=0.5,
    )

    return (ln,)


print("Writing video")
ani = FuncAnimation(
    fig, update, frames=n, init_func=init, blit=True, interval=100, repeat=False
)
writervideo = animation.FFMpegWriter(fps=10)
ani.save(filename="sample_oct.mp4", writer=writervideo)
plt.close()


"""
Ellipse drawing code. This is for the sparse visualization
"""
ells = [
    Ellipse(
        xy=(x[i], y[i]),
        width=width[i],
        height=height[i],
        angle=np.deg2rad(np.mod(theta[i], 2 * np.pi)),
    )
    for i in np.arange(start=0, stop=n, step=floor(n / 20))
]
