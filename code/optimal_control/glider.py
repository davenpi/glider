"""
Contains methods to solve the swing time-optimal control problem. From Petur.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import csv
import model_equations as me
from solvers import collocation_solver


def glider(N: int):
    """
    Implement the glider dynamics and define the control problem.

    Parameters
    ----------
    N : int
        Number of control intervals.
    """

    # Degree of interpolating polynomial
    d = 3

    # Control discretization
    N = N  # number of control intervals
    print("-----------------")
    print(f"The number of control intervals is {N}")
    print("-----------------")

    # Time horizon. This is a trick to make the final time a parameter for the
    # time optimal problem.
    T = 1.0
    tf = ca.SX.sym("tf")
    p = ca.vertcat(tf)

    # declare model variables
    u = ca.SX.sym("u")
    v = ca.SX.sym("v")
    w = ca.SX.sym("w")
    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    theta = ca.SX.sym("theta")
    beta = ca.SX.sym("beta")
    state = ca.vertcat(u, v, w, x, y, theta, beta)
    db_dt = ca.SX.sym("db_dt")

    # model equations
    dx_u = (
        (me.m0 + me.m2(beta)) * v * w
        - me.rho_f * me.gamma(u, v, w, beta) * v
        - np.pi
        * (me.rho_s - me.rho_f)
        * me.a_fun(beta)
        * me.b_fun(beta)
        * me.g
        * ca.sin(theta)
        - me.F(u, v, beta)
    ) / (me.m0 + me.m1(beta))
    dx_v = (
        -(me.m0 + me.m1(beta)) * u * w
        + me.rho_f * me.gamma(u, v, w, beta) * u
        - np.pi
        * (me.rho_s - me.rho_f)
        * me.a_fun(beta)
        * me.b_fun(beta)
        * me.g
        * ca.cos(theta)
        - me.G(u, v, beta)
    ) / (me.m0 + me.m2(beta))
    dx_w = tf * ((me.m1(beta) - me.m2(beta)) * u * v - me.M(w, beta)) / me.moi_tot(beta)
    dx_x = u * ca.cos(theta) - v * ca.sin(theta)
    dx_y = u * ca.sin(theta) + v * ca.cos(theta)
    dx_theta = w
    dx_beta = db_dt
    xdot = ca.vertcat(
        tf * dx_u,
        tf * dx_v,
        tf * dx_w,
        tf * dx_x,
        tf * dx_y,
        tf * dx_theta,
        tf * dx_beta,
    )

    # Objective term. The thing to be minimized by the controller.
    L = -(x**2) + db_dt**2
    L2 = tf

    # Define the casadi function we will pass to the solver.
    f = ca.Function(
        "f", [state, db_dt, p], [xdot, L], ["state", "db_dt", "p"], ["xdot", "L"]
    )
    f2 = ca.Function(
        "f", [state, db_dt, p], [xdot, L2], ["state", "db_dt", "p"], ["xdot", "L"]
    )

    # initial state
    x0 = [0.1, 0.1, 0, 0, 0, 0, 1]

    # Final state
    y_f = -100
    eq1 = y - y_f
    x_f = 121
    eq2 = x - x_f
    eq = ca.vertcat(eq1, eq2)

    xf_eq = ca.Function("xf_eq", [state], [eq], ["state"], ["eq"])

    # State Constraints
    x_lb = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.1]
    x_ub = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 10]

    # Control bounds
    u_lb = -1.0
    u_ub = 1.0

    # Parameter bounds and initial guess
    tf_guess = 17.0
    p_lb = [tf_guess - 15]
    p_ub = [tf_guess + 150]
    p_lb2 = [5.0]
    p_ub2 = [180.0]
    p0 = [tf_guess]

    # Open the file in binary mode
    with open("opt_guess_with_beta_dot.pkl", "rb") as file:
        # Call load method to deserialze
        opt_guess_bd = pickle.load(file)

    x_opt, u_opt, opt_guess, sol = collocation_solver(
        f,
        x0,
        x_lb,
        x_ub,
        N,
        T,
        u_lb=u_lb,
        u_ub=u_ub,
        p0=p0,
        p_lb=p_lb,
        p_ub=p_ub,
        d=d,
        xf_eq=xf_eq,
        opt_guess=opt_guess_bd,
    )

    x_opt2, u_opt2, _, _ = collocation_solver(
        f2,
        x0,
        x_lb,
        x_ub,
        N,
        T,
        xf_eq=xf_eq,
        u_lb=u_lb,
        u_ub=u_ub,
        p0=p0,
        p_lb=p_lb2,
        p_ub=p_ub2,
        opt_guess=opt_guess,
        d=d,
    )
    # Plot the result
    # tgrid = np.linspace(0, T, N + 1)
    # plt.plot(tgrid, x_opt2[3])
    # plt.plot(tgrid, x_opt2[4])
    # plt.step(tgrid, np.append(np.nan, u_opt[0]), "-.")
    # plt.legend(["x", "y", "db_dt"])
    # plt.grid()
    # plt.show()
    return x_opt2, u_opt2, opt_guess, sol


# import casadi as cas


# def swing_test(**kwargs):
#     # Problem parameters
#     theta_0 = np.pi / 4
#     dl_max = 0.05
#     u_max = 0.1
#     theta_f = np.pi

#     # Degree of interpolating polynomial
#     d = 3
#     # Time horizon
#     # T = 30.60
#     T = 1.0

#     tf = cas.SX.sym("tf")  # total time
#     p = cas.vertcat(tf)

#     # Declare model variables
#     x_theta = cas.SX.sym("θ")
#     x_p = cas.SX.sym("p")
#     x_l = cas.SX.sym("l")
#     x = cas.vertcat(x_theta, x_p, x_l)
#     u = cas.SX.sym("u")

#     # Model equations
#     dx_theta = x_p / (x_l**2)
#     dx_p = -x_l * cas.sin(x_theta)
#     dx_l = u_max * u
#     # xdot = cas.vertcat(dx_theta, dx_p, dx_l)
#     xdot = cas.vertcat(tf * dx_theta, tf * dx_p, tf * dx_l)
#     # xdot = cas.vertcat(x_p/(x_l**2), -x_l*cas.sin(x_theta), u_max*u)

#     # Objective term
#     L = -0.25 / x_l * (0.5 * (x_p / x_l) ** 2 - x_l * cas.cos(x_theta))
#     L2 = tf

#     # Continuous time dynamics
#     # p_dummy = cas.SX.sym('p_dummy', 0)
#     # f = cas.Function('f', [x, u, p_dummy], [xdot, L], ['x', 'u', 'p_dummy'], ['xdot', 'L'])
#     f = cas.Function("f", [x, u, p], [xdot, L], ["x", "u", "p"], ["xdot", "L"])
#     f2 = cas.Function("f2", [x, u, p], [xdot, L2], ["x", "u", "p"], ["xdot", "L2"])

#     # Initial state
#     x0 = [theta_0, 0.0, 1.0 + dl_max]

#     # Final state
#     eq = x_theta - theta_f
#     xf_eq = cas.Function("xf_eq", [x], [eq], ["x"], ["eq"])

#     # State constraints
#     x_lb = [-np.inf, -np.inf, 1.0 - dl_max]
#     x_ub = [np.inf, np.inf, 1.0 + dl_max]

#     # Control bounds
#     u_lb = -1.0
#     u_ub = 1.0

#     # Parameter bounds and initial guess
#     tf_guess = 30.6
#     p_lb = [tf_guess]
#     p_ub = [tf_guess]
#     p_lb2 = [10.0]
#     p_ub2 = [40.0]
#     p0 = [tf_guess]

#     # Control discretization
#     N = 300  # number of control intervals

#     x_opt, u_opt, opt_guess, _ = collocation_solver(
#         f,
#         x0,
#         x_lb,
#         x_ub,
#         N,
#         T,
#         u_lb=u_lb,
#         u_ub=u_ub,
#         p0=p0,
#         p_lb=p_lb,
#         p_ub=p_ub,
#         d=d,
#         **kwargs
#     )

#     x_opt2, u_opt2, _, _ = collocation_solver(
#         f2,
#         x0,
#         x_lb,
#         x_ub,
#         N,
#         T,
#         xf_eq=xf_eq,
#         u_lb=u_lb,
#         u_ub=u_ub,
#         p0=p0,
#         p_lb=p_lb2,
#         p_ub=p_ub2,
#         opt_guess=opt_guess,
#         d=d,
#         **kwargs
#     )

#     print(x_opt[0][-1] / np.pi)
#     print(x_opt2[0][-1] / np.pi)

#     # Plot the result
#     tgrid = np.linspace(0, T, N + 1)
#     # plt.plot(x_opt[0], x_opt[1])
#     # plt.legend(['x1','x2'])
#     plt.plot(tgrid, x_opt[0])
#     plt.plot(tgrid, x_opt2[0], "--")
#     plt.step(tgrid, np.append(np.nan, u_opt[0]), "-.")
#     plt.step(tgrid, np.append(np.nan, u_opt2[0]), "-.")
#     # plt.plot(tgrid, x_opt[2], '--')
#     # plt.xlabel('t')
#     # plt.legend(['x1','u'])
#     plt.grid()
#     plt.show()

#     return


if __name__ == "__main__":
    glider()
