"""
Implement the model equations.
"""
import numpy as np
import casadi as ca

# Define some global constants
CT = 1.2
CR = np.pi
A = 1.4
B = 1.0
mu = 0.2
nu = 0.2
g = 9.81
rho_f = 1
rho_s = 5
vol = 0.1  # actually just a*b
m0 = np.pi * rho_s * vol  # mass.


def moi(beta: float) -> float:
    """
    Gives the non dimensional moment of inertia.

    Equation 2.13 in Paoletti/Mahadevan. I = beta*rho_s/rho_f

    Parameters
    ----------
    beta : float
        Aspect ratio.

    Returns
    -------
    moi : float
        Non dimensional moment of inertia.
    """
    moi = beta * rho_s / rho_f
    return moi


def speed(u: float, v: float) -> float:
    """
    Compute the magnitude of the instantaneous velocity.

    Parameters
    ----------
    u : float
        Speed along longitudinal axis of cylinder.
    v : float
        Speed along vertical axis of cylinder.

    Returns
    -------
    speed : float
        Magnitude of instantaneous velocity.
    """
    speed_sq = u**2 + v**2
    speed = ca.sqrt(speed_sq)
    return speed


def F(u: float, v: float) -> float:
    """
    Return the value of the fluid force affecting longitudinal motion.

    This function computes the F given by equation 2.21 in the
    Paoletti/Mahadevan paper:

        F = 1/pi[A - B(u^2 - v^2)/(sqrt(u^2 + v^2))]*sqrt(u^2+v^2)*u

    Parameters
    ----------
    u : float
        Speed along longitudinal axis of cylinder.
    v : float
        Speed along vertical axis of cylinder.
    beta : float
        Aspect ratio of cylinder.

    Returns
    -------
    F : float
        Value of fluid force affecting the longitudinal motion.
    """
    F = 1 / np.pi * (A - B * (u**2 - v**2) / speed(u, v) ** 2) * speed(u, v) * u
    return F


def G(u: float, v: float) -> float:
    """
    Return the value of the fluid force affecting vertical motion.

    This function computes the G given by equation 2.22 in the
    Paoletti/Mahadevan paper:

        G = 1/pi(A - B*(u^2 - v^2)/(u^2 + v^2))*sqrt(u^2 + v^2)*v

    Parameters
    ----------
    u : float
        Speed along longitudinal axis of cylinder.
    v : float
        Speed along vertical axis of cylinder.
    beta : float
        Aspect ratio of cylinder.

    Returns
    -------
    G : float
        Value of fluid force affecting the vertical motion.
    """
    G = 1 / np.pi * (A - B * (u**2 - v**2) / speed(u, v) ** 2) * speed(u, v) * v
    return G


def gamma(u: float, v: float, w: float) -> float:
    """
    Compute function describing the circulation around the body.

    Use equation 2.20 in Paoletti/Mahadevan paper:

        gamma = 2/pi[(-C_t)*u*v/(sqrt(u^2+v^2)) + (2C_r)w]

    Parameters
    ----------
    u : float
        Speed along longitudinal axis of cylinder.
    v : float
        Speed along vertical axis of cylinder.
    w : float
        Angular velocity of cylinder.

    Returns
    -------
    gamma : float
        Value of circulation around the body at a given time.
    """
    gamma = 2 / np.pi * (-CT * u * v / speed(u, v) + CR * w)
    return gamma


def M(w: float) -> float:
    """
    Compute the fluid torque on the body.

    Using eqn (2.12) in Paoletti and Mahadevan we compute the
    effective fluid torque on the solid body given the instantaneous
    value of the parameters.

    M =  pi*rho_f*a^4[(V/L)*mu + nu*abs(w)]*w

    In the RL follow on to the original paper they have M = 0.2*M. I need
    to figure out which one is correct.

    Parameters
    ----------
    w : float
        Angular velocity of cylinder.
    beta : float
        Aspect ratio of the cylinder.

    Returns
    -------
    M : float
        Value of the fluid torque given the current parameters.
    """
    M = (mu + nu * ca.fabs(w)) * w
    return M
