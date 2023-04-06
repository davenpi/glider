"""
Contains methods to solve the optimal control problem. Borrowed a lot from Petur.
"""
import pickle
import numpy as np
import casadi as ca
import model_equations as me
from solvers import collocation_solver


def optimal_glider(
    N: int,
    use_upsampled_prior: bool,
    y_f: float,
    x_f: float,
    energy_optimal: bool = False,
    opt_guess=None,
):
    """
    Implement the glider dynamics and define the control problem.

    The reason for many of the extra boolean arguments is because I need to
    solve a different version of the control problem to create an initial guess
    for the time optimal problem. The boolean flags are used to indicate which
    problem I am solving and whether or not I want to use prior solutions as an
    initial guess.
    Also x_f is determined during the initial guess building phase. x_f is
    given by the maximum x that is reached when solving the first optimization
    problem.

    Parameters
    ----------
    N : int
        Number of control intervals.
    use_upsampled_prior : bool
        Passed directly to the collocation solver. This argument is True if
        I want to use a prior solution I constructed as an initial guess. In
        particular these prior solutions are the upsampled prior solutions from
        previous runs.
    y_f : float
        The final y position we are aiming for.
    x_f : float
        The final x position we are aiming for. Set to 0 if we are not
        including a final x target.
    energy_optimal: bool
        True if we want to solve the energy optimal problem.
    """
    # Degree of interpolating polynomial
    d = 3

    # Control discretization
    N = N  # number of control intervals

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
    if energy_optimal:
        L = db_dt**2
    else:
        L = tf

    # Define the casadi function we will pass to the solver.
    f = ca.Function(
        "f", [state, db_dt, p], [xdot, L], ["state", "db_dt", "p"], ["xdot", "L"]
    )

    # initial state
    x0 = [0.1, 0.1, 0, 0, 0, 0, 1]

    # Final state
    y_f = y_f
    eq1 = y - y_f
    eq2 = x - x_f
    # eq3 = theta - np.pi / 4
    eq = ca.vertcat(eq1, eq2)

    xf_eq = ca.Function("xf_eq", [state], [eq], ["state"], ["eq"])

    # State Constraints
    x_lb = [-np.inf, -np.inf, -np.inf, -np.inf, y_f, -np.inf, 0.1]
    x_ub = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 10]

    # Control bounds
    u_lb = -1.0
    u_ub = 1.0

    # Parameter bounds and initial guess
    tf_guess = 17.0
    p_lb = [1.0]
    p_ub = [180.0]
    p0 = [tf_guess]

    x_opt, u_opt, sol_x, sol = collocation_solver(
        f,
        x0,
        x_lb,
        x_ub,
        N,
        T,
        xf_eq=xf_eq,
        u_lb=u_lb,
        u_ub=u_ub,
        p0=p0,
        p_lb=p_lb,
        p_ub=p_ub,
        opt_guess=opt_guess,
        use_upsampled_prior=use_upsampled_prior,
        d=d,
    )
    return x_opt, u_opt, sol_x, sol


if __name__ == "__main__":
    optimal_glider()
