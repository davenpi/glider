"""
Contains methods to solve the swing time-optimal control problem. From Petur.
"""
import pickle
import numpy as np
import casadi as ca
import model_equations as me
from solvers import collocation_solver


def glider(
    N: int,
    use_upsampled_prior: bool,
    y_f: float,
    x_f: float,
    solve_2nd_optim: bool,
    using_opt_guess: bool,
    energy_optimal: bool,
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
    solve_2nd_optim : bool
        Flag that is true if I am interested in solving the time optimal
        problem or the energy optimal problem.
    y_f : float
        The final y position we are aiming for.
    x_f : float
        The final x position we are aiming for. Set to 0 if we are not
        including a final x target.
    using_opt_guess: bool
        True if we are using an optimal x returned by the solver for an
        initial guess.
    energy_optimal: bool
        True if we want to solve the energy optimal problem.
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
    dx_w = ((me.m1(beta) - me.m2(beta)) * u * v - me.M(w, beta)) / me.moi_tot(beta)
    dx_x = u * ca.cos(theta) - v * ca.sin(theta)
    dx_y = u * ca.sin(theta) + v * ca.cos(theta)
    dx_theta = w
    dx_beta = -(beta - 1) + db_dt
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
    L = db_dt**2
    if energy_optimal:
        L2 = db_dt**2
        print(L2)
    else:
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
    y_f = y_f
    eq1 = y - y_f
    if x_f == 0:
        eq = eq1
    else:
        eq2 = x - x_f
        eq = ca.vertcat(eq1, eq2)

    # Final state
    eq1 = y - y_f
    if x_f == 0:
        eq = eq1
    else:
        eq2 = x - x_f
        eq3 = ca.fmod(theta, 2 * np.pi) - np.pi / 4
        eq4 = beta - 1
        eq = ca.vertcat(eq1, eq2, eq3, eq4)

    xf_eq = ca.Function("xf_eq", [state], [eq], ["state"], ["eq"])

    # State Constraints
    x_lb = [-np.inf, -np.inf, -np.inf, -np.inf, y_f, -np.inf, 0.1]
    x_ub = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 10]

    # Control bounds
    u_ub = 5.0
    u_lb = -u_ub

    # Parameter bounds and initial guess
    tf_guess = 17.0
    p_lb = [1]
    p_ub = [180]
    p_lb2 = [1.0]
    p_ub2 = [180.0]
    p0 = [tf_guess]

    # Open the file in binary mode
    if using_opt_guess:
        with open("opt_guess_with_dim.pkl", "rb") as file:
            # Call load method to deserialze
            opt_guess_bd = pickle.load(file)
    else:
        opt_guess_bd = None

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
        use_upsampled_prior=use_upsampled_prior,
        opt_guess=opt_guess_bd,
    )

    if solve_2nd_optim:
        print("Solving the 2nd optimzation problem \n")
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
            use_upsampled_prior=False,
            d=d,
        )
        return x_opt2, u_opt2, _, _
    else:
        return x_opt, u_opt, opt_guess, sol


if __name__ == "__main__":
    glider()
