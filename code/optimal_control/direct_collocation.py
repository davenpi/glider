import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import model_equations as me

# Degree of interpolating polynomial
d = 3

# Get collocation points
tau_root = np.append(0, ca.collocation_points(d, "legendre"))

# Coefficients of the collocation equation
C = np.zeros((d + 1, d + 1))

# Coefficients of the continuity equation
D = np.zeros(d + 1)

# Coefficients of the quadrature function
B = np.zeros(d + 1)

# Construct polynomial basi
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


# size of state
state_dim = 7
# time horizon
T = 1.0
tf = ca.MX.sym("tf")
p = ca.vertcat(tf)
# number of control intervals
N = 90
h = T / N

# declare model variables
u = ca.MX.sym("u")
v = ca.MX.sym("v")
w = ca.MX.sym("w")
x = ca.MX.sym("x")
y = ca.MX.sym("y")
theta = ca.MX.sym("theta")
beta = ca.MX.sym("beta")
state = ca.vertcat(u, v, w, x, y, theta, beta)
db_dt = ca.MX.sym("db_dt")

# dynamical equations
xdot = ca.vertcat(
    tf
    * (
        (me.m0 + me.m2(beta)) * v * w
        - me.rho_f * me.gamma(u, v, w, beta) * v
        - np.pi
        * (me.rho_s - me.rho_f)
        * me.a_fun(beta)
        * me.b_fun(beta)
        * me.g
        * ca.sin(theta)
        - me.F(u, v, beta)
    )
    / (me.m0 + me.m1(beta)),
    tf
    * (
        -(me.m0 + me.m1(beta)) * u * w
        + me.rho_f * me.gamma(u, v, w, beta) * u
        - np.pi
        * (me.rho_s - me.rho_f)
        * me.a_fun(beta)
        * me.b_fun(beta)
        * me.g
        * ca.cos(theta)
        - me.G(u, v, beta)
    )
    / (me.m0 + me.m2(beta)),
    tf * ((me.m1(beta) - me.m2(beta)) * u * v - me.M(w, beta)) / me.moi_tot(beta),
    tf * (u * ca.cos(theta) - v * ca.sin(theta)),
    tf * (u * ca.sin(theta) + v * ca.cos(theta)),
    tf * w,
    tf * db_dt,
)

# objective term
L = x**2 + 2 * db_dt**2

# initial state and bounds
x0 = [0.1, 0.1, 0, 0, 0, 0, 1]
x_lb = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.1]
x_ub = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 10]

# continuous time dynamics
f = ca.Function(
    "f", [state, db_dt, p], [xdot, L], ["state", "db_dt", "p"], ["xdot", "L"]
)

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

p_lb = [10.0]
p_ub = [20.0]
p0 = [17.0]
# NLP Variables for the parameters to optimize
P = ca.MX.sym("P", 1)  # one parameter to optimize over
w.append(P)
lbw.append(p_ub)  # ending it at a given time for now
ubw.append(p_ub)
w0.append(p0)

# "Lift" initial conditions
Xk = ca.MX.sym("X0", state_dim)
w.append(Xk)
lbw.append(x0)
ubw.append(x0)
w0.append(x0)
x_plot.append(Xk)


# Formulate the NLP
u_ub = 0.5
u_lb = -0.5
u0 = 0.0

for k in range(N):
    # New NLP variable for the control
    Uk = ca.MX.sym("U_" + str(k))
    w.append(Uk)
    lbw.append([u_lb])
    ubw.append([u_ub])
    w0.append([u0])  # initial control
    u_plot.append(Uk)

    # State at collocation points
    Xc = []
    for j in range(d):
        Xkj = ca.MX.sym("X_" + str(k) + "_" + str(j), state_dim)
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
        lbg.append([0] * state_dim)
        ubg.append([0] * state_dim)

        # Add contribution to the end state
        Xk_end = Xk_end + D[j] * Xc[j - 1]

        # Add contribution to quadrature function
        J = J + B[j] * qj * h

    # New NLP variable for state at end of interval
    Xk = ca.MX.sym("X_" + str(k + 1), state_dim)
    w.append(Xk)
    lbw.append([-np.inf, -np.inf, -np.inf, -np.inf, -300, -np.inf, 0.1])
    ubw.append([np.inf, np.inf, np.inf, np.inf, 0, np.inf, 10])
    w0.append(x0)
    x_plot.append(Xk)

    # Add equality constraint
    g.append(Xk_end - Xk)
    lbg.append([0] * state_dim)
    ubg.append([0] * state_dim)

# Concatenate vectors
w = ca.vertcat(*w)
g = ca.vertcat(*g)
x_plot = ca.horzcat(*x_plot)
u_plot = ca.horzcat(*u_plot)
w0 = np.concatenate(w0)
lbw = np.concatenate(lbw)
ubw = np.concatenate(ubw)
lbg = np.concatenate(lbg)
ubg = np.concatenate(ubg)

# Create an NLP solver
prob = {"f": J, "x": w, "g": g}
solver = ca.nlpsol("solver", "ipopt", prob)

# Function to get x and u trajectories from w
trajectories = ca.Function(
    "trajectories", [w], [x_plot, u_plot], ["w"], ["state", "db_dt"]
)

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
x_opt, u_opt = trajectories(sol["x"])
x_opt = x_opt.full()  # to numpy array
u_opt = u_opt.full()  # to numpy array

# Plot the result
tgrid = np.linspace(0, T, N + 1)
plt.figure(1)
plt.clf()
plt.plot(tgrid, x_opt[3], "--")
plt.plot(tgrid, x_opt[4], "-")
plt.step(tgrid, np.append(np.nan, u_opt[0]), "-.")
plt.xlabel("t")
plt.legend(["x", "y", "db_dt"])
plt.grid()
plt.show()
