"""
This file creates movies showing glider trajectories over time.
"""
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import pickle
from math import floor


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--opt_path", help="Path to optimal arrays", type=str)

args = parser.parse_args()
path = args.opt_path

x_opt = np.load(path + "/x_opt.npy")
u_opt = np.load(path + "/u_opt.npy")
x = np.array(x_opt[3])
x_min, x_max = x.min(), x.max()
y = np.array(x_opt[4])
y_min, y_max = y.min(), y.max()
theta = np.array(x_opt[5])
beta = np.array(x_opt[6])
n = beta.size
width = np.sqrt(1 / beta)
height = np.sqrt(beta)

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
        width=width[frame],
        height=height[frame],
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
        dx=np.cos(theta[frame]),
        dy=np.sin(theta[frame]),
        width=0.1,
    )

    return (ln,)


print("Writing video")
ani = FuncAnimation(
    fig, update, frames=n, init_func=init, blit=True, interval=20, repeat=False
)
writervideo = animation.FFMpegWriter(fps=30)
ani.save(filename=path + "/sample_oct.mp4", writer=writervideo)
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


print("Plotting logged data")
fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(10, 8))
fig.text(0.5, 0.04, "Time", ha="center")
fig.suptitle("Logged info from trajectory", fontsize=20)
ax[0, 0].plot(theta)
ax[0, 0].set_ylabel(r"$\theta$")
ax[0, 1].plot(x)
ax[0, 1].set_ylabel("X")
ax[1, 0].plot(y)
ax[1, 0].set_ylabel("Y")
ax[1, 1].plot(beta)
ax[1, 1].plot(u_opt[0])
ax[1, 1].set_ylabel(r"$\beta$")
ax[1, 1].legend(("beta", "beta_dot"))
v = x_opt[1]
ax[2, 0].plot(v)
ax[2, 0].set_ylabel("V")
u = x_opt[0]
ax[2, 1].plot(u)
ax[2, 1].set_ylabel("U")
w = x_opt[2]
ax[3, 0].plot(w)
ax[3, 0].set_ylabel("W")
# trajectory
ax[3, 1].scatter(x, y)
ax[3, 1].set_xlabel("X")
ax[3, 1].set_ylabel("Y")

plt.savefig(path + "/logged_info.png")
plt.close()
