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

parser = argparse.ArgumentParser(description="Process name")
parser.add_argument("fname", type=str)
args = parser.parse_args()

with open("history.pkl", "rb") as f:
    history = pickle.load(f)

x = np.array(history["x"])
x_min, x_max = x.min(), x.max()
y = np.array(history["y"])
y_min, y_max = y.min(), y.max()

theta = np.array(history["theta"])
beta = np.array(history["beta"])
n = len(history["beta"])
width = np.ones(shape=n)
height = beta * width


"""
First let me just try to animate trajectory of the glider using the position
of its geometric center. I will try to add in drawings of the ellipse later.
"""

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
    ax.set_ylim(y_min, y_max + pad)
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
writervideo = animation.FFMpegWriter(fps=20)
ani.save(args.fname, writer=writervideo)
plt.close()
plt.show()


"""
Ellipse drawing code
"""
# ells = [
#     Ellipse(
#         xy=(x[i], y[i]),
#         width=width[i],
#         height=height[i],
#         angle=theta[i],
#     )
#     for i in range(n)
# ]

# fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})
# for e in ells:
#     ax.add_artist(e)
#     e.set_clip_box(ax.bbox)
#     e.set_alpha(np.random.rand())
#     e.set_facecolor(np.random.rand(3))

# ax.set_xlim(x.min(), x.max())
# ax.set_ylim(y.min(), y.max())

# plt.show()
