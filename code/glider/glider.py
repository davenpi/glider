"""
Here I implement the glider class as a gym environment.
"""
import gym
import numpy as np
from scipy.integrate import solve_ivp


# Define some global constants
CT = 1.2
CR = np.pi
A = 1.4
B = 1.0
mu = 0.2
nu = 0.2
g = 9.81


class Glider(gym.Env):
    def __init__(
        self,
        rho_s: float = 2,
        rho_f: float = 1,
        beta0: float = 1,
        u0: float = -0.1,
        v0: float = 0.25,
        w0: float = 0.0,
        x0: float = 0,
        y0: float = 0,
        theta0: float = np.pi / 8,
        terminal_y: float = -20,
        target_x: float = 10,
        beta_min: float = 0.1,
        ellipse_volume: float = 1,
    ):
        """
        This is the main glider class.
        """
        super(Glider, self).__init__()
        self.t = 0
        self.dt = 0.1
        self.rho_s = rho_s
        self.mass = 1
        self.rho_f = rho_f
        self.vol = ellipse_volume
        self.u0 = u0
        self.v0 = v0
        self.w0 = w0
        self.x0 = x0
        self.y0 = y0
        self.theta0 = theta0
        self.beta0 = beta0
        self.beta = [beta0]
        self.beta_max = 1 / beta_min
        self.beta_min = beta_min
        self.u = [u0]
        self.v = [v0]
        self.w = [w0]
        self.x = [x0]
        self.y = [y0]
        self.theta = [theta0]
        self.ang_limit = np.pi / 2
        self.terminal_y = terminal_y
        self.t_hist = [0]
        self.t_max = 1000 * self.dt
        self.u_max = 8  # guess
        self.v_max = 4  # guess
        self.max_speed = 10
        self.max_x = 4 * target_x
        self.max_y = np.abs(
            self.terminal_y
        )  # should center the [terminal_y, 0] interval
        self.observation_space = gym.spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1, -1]),
            high=np.array([1, 1, 1, 1, 1, 1, 1]),
        )  # state is (u, v, w, x, y, theta, beta)
        self.action_space = gym.spaces.Discrete(3)
        self.target_x = target_x
        self.target_theta = np.pi / 4
        self.lookup_dict = {0: 0, 1: 5, 2: -5}

    def a(self, beta: float) -> float:
        """
        Compute a given a beta.

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

        a = np.sqrt(self.vol / beta)
        return a

    def b(self, beta: float) -> float:
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
        b = np.sqrt(beta * self.vol)
        return b

    def V(self, beta: float) -> float:
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
        V = np.sqrt((self.rho_s / self.rho_f - 1) * g * self.b(beta))
        return V

    def M(self, w: float, beta: float) -> float:
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
        L = self.a(beta)
        M = (
            (np.pi * self.rho_f * self.a(beta) ** 4)
            * (mu * self.V(beta) / L + nu * np.abs(w))
            * w
        )
        return M

    def G(self, u: float, v: float, beta: float) -> float:
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
            self.rho_f
            * self.a(beta)
            * (A - B * (u**2 - v**2) / self.speed(u, v) ** 2)
            * self.speed(u, v)
            * v
        )
        return G

    def F(self, u: float, v: float, beta: float) -> float:
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
            self.rho_f
            * self.a(beta)
            * (A - B * (u**2 - v**2) / self.speed(u, v) ** 2)
            * self.speed(u, v)
            * u
        )
        return F

    def speed(self, u: float, v: float) -> float:
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
        speed = np.sqrt(speed_sq)
        return speed

    def gamma(self, u: float, v: float, w: float, beta: float) -> float:
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
            -2 * CT * self.a(beta) * u * v / self.speed(u, v)
            + 2 * CR * (self.a(beta) ** 2) * w
        )
        return gamma

    def m0(self, beta: float) -> float:
        """
        Compute basic mass of ellipse.

        m = pi*rho_s*a*b

        Parameters
        ----------
        beta : float
            Cylinder aspect ratio.

        Returns
        -------
        m0 : float
            Mass of ellipse.
        """
        a = self.a(beta)
        b = self.b(beta)
        m0 = np.pi * self.rho_s * a * b
        return m0

    def m1(self, beta: float) -> float:
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
        b = self.b(beta)
        m1 = np.pi * self.rho_f * b**2
        return m1

    def m2(self, beta: float) -> float:
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
        a = self.a(beta)
        m2 = np.pi * self.rho_f * a**2
        return m2

    def moi0(self, beta: float) -> float:
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
        a = self.a(beta)
        b = self.b(beta)
        I0 = 0.25 * np.pi * self.rho_s * a * b * (a**2 + b**2)
        return I0

    def moi_renorm(self, beta: float) -> float:
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
        a = self.a(beta)
        b = self.b(beta)
        Ia = 0.125 * np.pi * self.rho_f * (a**2 - b**2) ** 2
        return Ia

    def moi_tot(self, beta: float) -> float:
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
        I0 = self.moi0(beta)
        Ia = self.moi_renorm(beta)
        moi_tot = I0 + Ia
        return moi_tot

    def dynamical_eqns(self, t: float, s: np.ndarray, db_dt: float) -> np.ndarray:
        """
        Give the dynamics of the system at a given time.

        Input the time and the state and return the dynamics describing
        how the state will evolve over time. The state of the system is
        s = (u, v, w, x, y, \theta, beta). The dynamical equations are given
        by equations 2.1-2.6 in the Paoletti/Mahadevan paper plus the added
        beta dynamics.

        Parameters
        ----------
        t : float
            Current time.
        s : np.ndarray
            State of the system. (u, v, w, x, y, theta, beta)
        db_dot : float
            Beta dot.

        Returns
        -------
        ds_dt : np.ndarray
            Vector containing equations for evolution of state variables.
        """
        a = self.a(beta=s[6])
        b = self.b(beta=s[6])
        m0 = self.m0(beta=s[6])
        m1 = self.m1(beta=s[6])
        m2 = self.m2(beta=s[6])
        I_tot = self.moi_tot(beta=s[6])
        u_dot = (
            (m0 + m2) * s[1] * s[2]
            - self.rho_f * self.gamma(u=s[0], v=s[1], w=s[2], beta=s[6]) * s[1]
            - np.pi * (self.rho_s - self.rho_f) * a * b * g * np.sin(s[5])
            - self.F(u=s[0], v=s[1], beta=s[6])
        ) / (m0 + m1)
        v_dot = (
            -(m0 + m1) * s[0] * s[2]
            + self.rho_f * self.gamma(u=s[0], v=s[1], w=s[2], beta=s[6]) * s[0]
            - np.pi * (self.rho_s - self.rho_f) * a * b * g * np.cos(s[5])
            - self.G(u=s[0], v=s[1], beta=s[6])
        ) / (m0 + m2)
        w_dot = ((m1 - m2) * s[0] * s[1] - self.M(w=s[2], beta=s[6])) / (I_tot)
        x_dot = s[0] * np.cos(s[5]) - s[1] * np.sin(s[5])
        y_dot = s[0] * np.sin(s[5]) + s[1] * np.cos(s[5])
        theta_dot = s[2]
        beta_dot = db_dt
        s_dot = np.hstack((u_dot, v_dot, w_dot, x_dot, y_dot, theta_dot, beta_dot))
        return s_dot

    def update_state_history(self, solution_object) -> None:
        """
        Update the state history using the solution object found with solve_ivp

        I am assuming that the first element of the solution is the initial
        condition so I don't append that to the state. From the examples in
        the docs it seems to be the case that the first element in each
        solution array is just the initial conidtion so I think this is a fine
        approach. We also add here code to make sure we are logging the proper
        number of betas in the history so there are an equal amount of betas
        and other state variables.

        Parameters
        ----------
        solution_object : Bunch
            Bunch object defined in Scipy that holds the solution and times
            at which the solution is found. The solution is contained in the
            y property of the object.
        beta : float
            Value of aspect ratio for the duration of the current simulation
            step.
        """
        u = solution_object.y[0]
        self.u.extend(list(u[1:]))
        v = solution_object.y[1]
        self.v.extend(list(v[1:]))
        w = solution_object.y[2]
        self.w.extend(list(w[1:]))
        x = solution_object.y[3]
        self.x.extend(list(x[1:]))
        y = solution_object.y[4]
        self.y.extend(list(y[1:]))
        theta = solution_object.y[5]
        self.theta.extend(list(theta[1:]))
        beta = solution_object.y[6]
        self.beta.extend(list(beta[1:]))
        self.t_hist.extend(list(solution_object.t[1:]))
        self.t += self.dt

    def check_beta_bound(self, beta_dot: float) -> float:
        """
        Check to make sure that beta does not go outside of the bounds.

        Parameters
        ----------
        beta_dot : float
            Control input.

        Returns
        -------
        beta_dot : float
            Edited control if necessary.
        """
        if self.beta[-1] + self.dt * beta_dot > self.beta_max:
            beta_dot = (self.beta_max - self.beta[-1]) / self.dt
        elif self.beta[-1] + self.dt * beta_dot < self.beta_min:
            beta_dot = (self.beta_min - self.beta[-1]) / self.dt
        else:
            beta_dot = beta_dot
        return beta_dot

    def forward(self, beta_dot: float) -> None:
        """
        Integrating the dynamics forward in time.

        The dynamics are given in equations 2.14-2.19 in the
        Paoletti/Mahadevan paper. We use the scipy ODE solver to integrate
        them forward in time. The scipy solver takes equations of the form
            ds/dt = f(t, s)
            s(t0) = s0
        and integrates them from a given initial to final time.
        Alongside integrating the equations forward in time we need to make
        sure we update the history of the state variables. Since we want to
        develop a controller which chooses a value of beta to determine glide
        performance, we need to accept a value of beta as an input to this
        function. We have to update that value of

        Parameters
        ----------
        beta_dot : float
            Rate of change of cylinder aspect ratio.

        Returns
        -------
        sol_object : bunch object with relevant fields listed below
            t : np.ndarry (n_points,)
                Time points where solution was found.
            y : np.ndarray (n, n_points)
                Value of state at the n_points
        """
        beta_dot = self.check_beta_bound(beta_dot=beta_dot)
        sol_object = solve_ivp(
            fun=self.dynamical_eqns,
            t_span=[self.t, self.t + self.dt],
            y0=[
                self.u[-1],
                self.v[-1],
                self.w[-1],
                self.x[-1],
                self.y[-1],
                self.theta[-1],
                self.beta[-1],
            ],
            args=[beta_dot],
        )
        self.update_state_history(solution_object=sol_object)

    def check_angle_bound(self, angle: float) -> bool:
        """
        Check the state of the system and compute the requisite reward.

        MAYBE I SHOULD CHECK THIS INSIDE THE STATE WITH EVENTS. That's
        definitely a better solution. I will learn how to do that. I don't
        think there will be a large difference though because the time step is
        so small that there are only a couple of angles in between the
        observed angles. I also suspect that there is not significant short
        time oscillation in the angle that would make it go out of bounds in
        the middle of a time step but back in bounds by the end.

        Parameters
        ----------
        angle : float
            Current angle of the cylinder.

        Returns
        -------
        out_of_bounds : bool
            True if the angle is outside of bounds.
        """
        if np.abs(angle) > self.ang_limit:
            out_of_bounds = True
        else:
            out_of_bounds = False
        return out_of_bounds

    def check_hit_ground(self, y: float) -> bool:
        """
        Check if the cylinder reached the ground.

        AGAIN. DO THIS WITH EVENTS IN THE LONG RUN. For the same reasons as
        before I don't think it will be a huge problem.

        Parameters
        ----------
        y : float
            Current y position.
        Returns
        -------
        hit_ground : bool
            True if hit ground. False otherwise.
        """
        if y <= self.terminal_y:
            hit_ground = True
        else:
            hit_ground = False
        return hit_ground

    def action_lookup(self, action: int) -> float:
        """
        Lookup the action value according to a specific integer.

        Need to define an action dictionary because the gym Discrete aciton
        space only gives integer choices.

        Parameters
        ----------
        action : int
            Integer action chose by agent.

        Returns
        -------
        beta_dot : float
            Rate of change of aspect ratio according to lookup dictionary.
        """
        beta_dot = self.lookup_dict[action]
        return beta_dot

    def compute_reward(self) -> float:
        """
        Compute the reward the agent will normally receive.

        This is not the reward at all times because I have a diferent reward
        for when the agent flips.

        Paramters
        ---------
        None

        Returns
        -------
        reward : float
            Reward given to the RL agent.
        """
        reward = (
            -self.dt
            + np.abs(self.target_x - self.x[-2])
            - np.abs(self.target_x - self.x[-1])
        )
        return reward

    def scale_beta(self) -> float:
        """
        Return the scaled beta which is in the intereval [-1, 1]

        Parameters
        ----------

        Returns
        -------
        norm_beta : float
            Normalized beta value.
        """
        norm_beta = (-2 / (self.beta_min - self.beta_max)) * (
            self.beta[-1] - self.beta_max
        ) + 1
        return norm_beta

    def extract_observation(self) -> np.ndarray:
        """
        Return the state observation.

        Parameters
        ----------
        None

        Returns
        -------
        obs : np.ndarray
            Observation given to the agent.
        """
        # speed = np.sign(self.u[-1]) * self.speed(u=self.u[-1], v=self.v[-1])
        # norm_speed = speed / self.max_speed
        norm_u = self.u[-1] / self.u_max
        norm_v = self.v[-1] / self.v_max
        norm_w = self.w[-1] / (2 * self.ang_limit / self.dt)

        x_pos = self.x[-1]
        delta_x = x_pos - self.target_x
        norm_delta_x = delta_x / self.max_x
        y_pos = self.y[-1]
        delta_y = y_pos - self.terminal_y
        norm_delta_y = delta_y / np.abs(self.terminal_y)
        norm_angle = self.theta[-1] / self.ang_limit
        norm_beta = self.scale_beta()
        obs = np.array(
            [norm_u, norm_v, norm_w, norm_delta_x, norm_delta_y, norm_angle, norm_beta],
            dtype=np.float32,
        )

        return obs

    def step(self, action: float) -> tuple:
        """
        Classic step method which moves the environment forward one step in time.

        Parameters
        ----------
        action : float
            The action in this problem will be a choice of cylinder aspect
            ratio beta.

        Returns
        -------
        obs : np.ndarray (N,)
            N dimensional observation for the agent. For right now I will make
            the observation be the instantaneous speed and angle:
                obs = (speed, theta)
        reward : float
            Scalar reward the agent. We will positively reward long glides.
            We strongly penalize excessive tumbling beyond theta limits so
            that the agent doesn't exhibit this behavior. We may also want to
            reward gliding somewhere quickly or gliding to a specific spot.
        done : bool
            Whether or not the episode is done. We will end episodes when the
            cylinder has tipped too far, the time has run out, or it has
            reached the terminal y position.
        info : dict
            Extra information about the episode for logging purposes. Right
            now I will not include any extra logging info.
        """
        beta_dot = self.action_lookup(action)
        self.forward(beta_dot=beta_dot)
        obs = self.extract_observation()
        angle_out_of_bounds = self.check_angle_bound(angle=self.theta[-1])
        hit_ground = self.check_hit_ground(y=self.y[-1])
        # if angle_out_of_bounds:
        #     # large penalty for flipping and end episode
        #     reward = -1000
        #     done = True
        if hit_ground:
            reward = self.compute_reward()
            reward += 10 * (np.exp(-((self.x[-1] - self.target_x) ** 2)))
            # if self.t > 5:
            #     reward += 25 * (
            #         np.exp(-5 * ((np.abs(self.theta[-1]) - self.target_theta) ** 2))
            #     )
            done = True
        else:
            reward = self.compute_reward()
            done = False
        info = {}
        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        """
        Reset method for gym environment. Resets the system to the original.

        Make sure to reset all of the variables to their initial values. I can
        also experiment with random initial conditions in some range here at a
        later point. For now I will keep it fixed.
        """
        self.u = [self.u0]
        self.v = [self.v0]
        self.w = [self.w0]
        self.x = [self.x0]
        self.y = [self.y0]
        self.theta = [self.theta0]
        self.beta = [self.beta0]
        self.t_hist = [0]
        self.t = 0
        obs = self.extract_observation()
        return obs

    def render(self) -> None:
        pass
