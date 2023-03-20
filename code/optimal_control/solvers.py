"""
Contains methods to solve optimal control problem using a direct collocation method
"""

# The code is based on direct_collocation.py from the CasADi examples
# https://github.com/casadi/casadi/blob/master/docs/examples/python/direct_collocation.py

import os
import numpy as np
import matplotlib.pyplot as plt
import casadi as cas
import csv


def collocation_solver(
    f,
    x0,
    x_lb,
    x_ub,
    N,
    T=1.0,
    *,
    xf=None,
    xf_eq=None,
    u_lb=-1.0,
    u_ub=1.0,
    p0=None,
    p_lb=None,
    p_ub=None,
    opt_guess=None,
    d=3,
    ipopt_options=None,
    use_upsampled_prior=False,
):
    if ipopt_options is None:
        ipopt_options = {
            "ipopt.sb": "yes",
            "ipopt.print_level": 0,
            "print_time": False,
            "ipopt.max_iter": 100,
        }

    # Time step
    h = T / N
    print(f"The time step is {h}")

    # Dimensionality of the state, control and parameter vector
    n = f.size1_in(0)
    print(f"The dimension of the state is {n}")
    m = f.size1_in(1)
    print(f"The dimension of the control is {m}")
    n_p = f.size1_in(2)
    print(f"The number of parameters is {n_p}")
    print("---\n\n")

    if p_lb is None:
        p_lb = [-np.inf] * n_p
    if p_ub is None:
        p_ub = [np.inf] * n_p
    if p0 is None:
        p0 = [0.0] * n_p

    # Get collocation points
    tau_root = np.append(
        0, cas.collocation_points(d, "legendre")
    )  # can use radau. don't know it but something to try

    # Coefficients of the collocation equation
    C = np.zeros((d + 1, d + 1))

    # Coefficients of the continuity equation
    D = np.zeros(d + 1)

    # Coefficients of the quadrature function
    B = np.zeros(d + 1)

    # Construct polynomial basis
    for j in range(d + 1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d + 1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)

        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(d + 1):
            C[j, r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)

    # ----- Construct the NLP -----

    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g = []
    lbg = []
    ubg = []

    # For plotting x and u given w
    x_plot = []
    u_plot = []

    # NLP variables for the parameters to optimize
    P = cas.MX.sym("P", n_p)
    if n_p > 0:
        w.append(P)
        lbw.append(p_lb)
        ubw.append(p_ub)
        w0.append(p0)

    # "Lift" initial conditions
    Xk = cas.MX.sym("X0", n)
    w.append(Xk)
    lbw.append(x0)
    ubw.append(x0)
    w0.append(x0)
    x_plot.append(Xk)

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = cas.MX.sym("U_" + str(k), m)
        w.append(Uk)
        lbw.append([u_lb])
        ubw.append([u_ub])
        w0.append([0.0])
        u_plot.append(Uk)

        # State at collocation points
        Xc = []
        for j in range(d):
            Xkj = cas.MX.sym("X_" + str(k) + "_" + str(j), n)
            Xc.append(Xkj)
            w.append(Xkj)
            lbw.append(x_lb)
            ubw.append(x_ub)
            w0.append(x0)

        # Loop over collocation points
        Xk_end = D[0] * Xk
        for j in range(1, d + 1):
            # Expression for the state derivative at the collocation point
            xp = C[0, j] * Xk
            for r in range(d):
                xp = xp + C[r + 1, j] * Xc[r]

            # Append collocation equations
            fj, qj = f(Xc[j - 1], Uk, P)
            g.append(h * fj - xp)
            lbg.append([0] * n)
            ubg.append([0] * n)

            # Add contribution to the end state
            Xk_end = Xk_end + D[j] * Xc[j - 1]

            # Add contribution to quadrature function
            J = J + B[j] * qj * h

        # New NLP variable for state at end of interval
        Xk = cas.MX.sym("X_" + str(k + 1), n)
        w.append(Xk)
        lbw.append(x_lb)
        ubw.append(x_ub)
        w0.append(x0)
        x_plot.append(Xk)

        # Add equality constraint
        g.append(Xk_end - Xk)
        lbg.append([0] * n)
        ubg.append([0] * n)

    # - Terminal condition -
    if xf is not None:
        g.append(Xk)  # Xk is the final state
        lbg.append(xf)
        ubg.append(xf)
    elif xf_eq is not None:
        g.append(xf_eq(Xk))  # Xk is the final state
        n_eq = xf_eq.size1_out(0)
        lbg.append([0] * n_eq)
        ubg.append([0] * n_eq)

    # print("Before concatenation.\n")
    # print(f"w0 is {w0}\n")

    # Concatenate vectors
    w = cas.vertcat(*w)
    g = cas.vertcat(*g)
    x_plot = cas.horzcat(*x_plot)
    u_plot = cas.horzcat(*u_plot)
    w0 = np.concatenate(w0)
    lbw = np.concatenate(lbw)
    ubw = np.concatenate(ubw)
    lbg = np.concatenate(lbg)
    ubg = np.concatenate(ubg)
    # print("After concatenation.\n")
    # print(f"w0 is {w0}\n")

    # Create an NLP solver
    prob = {"f": J, "x": w, "g": g}
    # solver = cas.nlpsol("solver", "ipopt", prob, ipopt_options)
    solver = cas.nlpsol("solver", "ipopt", prob)

    # Function to get x and u trajectories from w
    trajectories = cas.Function(
        "trajectories", [w], [x_plot, u_plot], ["w"], ["x", "u"]
    )

    # Solve the NLP
    if opt_guess is not None:
        print("Using opt_guess")
        sol = solver(x0=opt_guess, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    else:
        if use_upsampled_prior:
            w0 = np.load("double_w0.npy")
            w0 = np.load("w0.npy")
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    x_opt, u_opt = trajectories(sol["x"])
    x_opt = x_opt.full()  # to numpy array
    u_opt = u_opt.full()  # to numpy array

    return x_opt, u_opt, sol["x"], sol
