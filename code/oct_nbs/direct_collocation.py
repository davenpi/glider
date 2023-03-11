from casadi import *
import matplotlib.pyplot as plt
import numpy as np
import model_equations as me

T = 17.0  # time horizon
N = 20  # number of control intervals

# declare model variables
u = MX.sym("u")
v = MX.sym("v")
w = MX.sym("w")
x = MX.sym("x")
y = MX.sym("y")
theta = MX.sym("theta")
beta = MX.sym("beta")
state = vertcat(u, v, w, x, y, theta, beta)
db_dt = MX.sym("db_dt")

# dynamical equations
xdot = vertcat(
    (
        (me.m0 + me.m2(beta)) * v * w
        - me.rho_f * me.gamma(u, v, w, beta) * v
        - np.pi * (me.rho_s - me.rho_f) * me.a(beta) * me.b(beta) * me.g * np.sin(theta)
        - me.F(u, v, beta)
    )
    / (me.m0 + me.m1(beta)),
    (
        -(me.m0 + me.m1(beta)) * u * w
        + me.rho_f * me.gamma(u, v, w, beta) * u
        - np.pi * (me.rho_s - me.rho_f) * me.a(beta) * me.b(beta) * me.g * np.cos(theta)
        - me.G(u, v, beta)
    )
    / (me.m0 + me.m2(beta)),
    ((me.m1(beta) - me.m2(beta)) * u * v - me.M(w, beta)) / me.moi_tot(beta),
    u * np.cos(theta) - v * np.sin(theta),
    u * np.sin(theta) + v * np.cos(theta),
    w,
    db_dt,
)
