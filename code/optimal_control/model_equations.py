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
rho_s = 2
vol = 1
m0 = np.pi * rho_s * vol  # mass.


def a_fun(beta: float) -> float:
    """
    Compute length of semi-major axis of ellipse.

    Since I am assuming the mass of the object m = pi*rho_s*a*b is fixed
    and that the density of the object is fixed, I need to update the size
    of the semi-major axis whenever beta is updated. Assuming a*b = vol we
    let
        a = sqrt(vol/beta)

    Parameters
    ----------
    beta : float
        Current value of the aspect ratio.

    Returns
    -------
    a : float
        Ellipse semi major axis length.
    """

    a = ca.sqrt(vol / beta)
    return a


def b_fun(beta: float) -> float:
    """
    Compute length of semi-minor axis of ellipse.

    Keeping a*b = vol a constant requires updating a and b as we update
    beta. We have a = sqrt(vol/beta) so that b = sqrt(beta*vol)

    Parameters
    ----------
    beta : float
        Cylinder aspect ratio.

    Returns
    -------
    b : float
        Cylinder semi-minor axis length.
    """
    b = ca.sqrt(beta * vol)
    return b


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


def F(u: float, v: float, beta: float) -> float:
    """
    Return the value of the fluid force affecting longitudinal motion.

    This function computes the F given by equation 2.10 in the
    Paoletti/Mahadevan paper:

        F = rho_f*a[A - B(u^2 - v^2)/(sqrt(u^2 + v^2))]*sqrt(u^2+v^2)*u

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
    F = (
        rho_f
        * a_fun(beta)
        * (A - B * (u**2 - v**2) / speed(u, v) ** 2)
        * speed(u, v)
        * u
    )
    return F


def G(u: float, v: float, beta: float) -> float:
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
    G = (
        rho_f
        * a_fun(beta)
        * (A - B * (u**2 - v**2) / speed(u, v) ** 2)
        * speed(u, v)
        * v
    )
    return G


def gamma(u: float, v: float, w: float, beta: float) -> float:
    """
    Compute function describing the circulation around the body.

    Use equation 2.9 in Paoletti/Mahadevan paper:

        gamma = (-2C_t)a*u*v/(sqrt(u^2+v^2)) + (2C_r)wa^2

    Parameters
    ----------
    u : float
        Speed along longitudinal axis of cylinder.
    v : float
        Speed along vertical axis of cylinder.
    w : float
        Angular velocity of cylinder.
    beta : float
        Cylinder aspect ratio.

    Returns
    -------
    gamma : float
        Value of circulation around the body at a given time.
    """
    gamma = (
        -2 * CT * a_fun(beta) * u * v / speed(u, v) + 2 * CR * (a_fun(beta) ** 2) * w
    )
    return gamma


def M(w: float, beta: float) -> float:
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
    L0 = a_fun(beta)
    M = (np.pi * rho_f * a_fun(beta) ** 4) * (mu * V_fun(beta) / L0 + nu * ca.fabs(w)) * w
    return M


def V_fun(beta: float) -> float:
    """
    Compute characteristic velocity scale given in Paoletti paper.

    V = sqrt((rho_s/rho_f - 1)*g*b) from page 492 of Paoletti paper.

    Parameters
    ----------
    beta : float
        Cylinder aspect ratio.

    Returns
    -------
    V : float
        Instantaneous value of the characteristic velocity.
    """
    V = np.sqrt(np.abs(rho_s / rho_f - 1) * g * b_fun(beta))
    return V


def m1(beta: float) -> float:
    """
    Compute added mass in u direction

    m1 = pi*rho_f*b^2

    Parameters
    ----------
    beta : float
        Cylinder aspect ratio.

    Returns
    -------
    m1 : float
        Added mass along u.
    """
    b = b_fun(beta)
    m1 = np.pi * rho_f * b**2
    return m1


def m2(beta: float) -> float:
    """
    Compute added mass along v.

    m2 = pi*rho_f*a^2

    Parameters
    ----------
    beta : float
        Cylinder aspect ratio.

    Returns
    -------
    m2 : Added mass along v.
    """
    a = a_fun(beta)
    m2 = np.pi * rho_f * a**2
    return m2


def moi0(beta: float) -> float:
    """
    Compute moment of inertia.


    Parameters
    ----------
    beta : float
        Cylinder aspect ratio.

    Returns
    -------
    I0 : float
        Moment of inertia.
    """
    a = a_fun(beta)
    b = b_fun(beta)
    I0 = 0.25 * m0 * (a**2 + b**2)
    return I0


def moi_renorm(beta: float) -> float:
    """
    Compute renormalization addition to the moment of inertia.

    Given in equation 2.8 of Paoletti.

    Parameters
    ----------
    beta : float
        Cylinder aspect ratio.

    Returns
    -------
    Ia : float
        Renormalization addition to the moment of inertia.
    """
    a = a_fun(beta)
    b = b_fun(beta)
    Ia = 0.125 * np.pi * rho_f * (a**2 - b**2) ** 2
    return Ia


def moi_tot(beta: float) -> float:
    """
    Compute the total inertia in equation 2.3 from eqn 2.7-2.8 in Paoletti.

    Add the regular moment to the renormalized moment.

        I_tot = I + I_a

    Parameters
    ----------
    beta : float
        Aspect ratio.

    Returns
    -------
    moi : float
        Non dimensional moment of inertia.
    """
    I0 = moi0(beta)
    Ia = moi_renorm(beta)
    moi_tot = I0 + Ia
    return moi_tot
