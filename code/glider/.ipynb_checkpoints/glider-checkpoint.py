import numpy as np
from scipy.integrate import solve_ivp


class Glider:
    def __init__(self, rho_s: float = 1, rho_f: float = 0.1) -> None:
        self.rho_s = rho_s
        self.rho_f = rho_f
        pass
