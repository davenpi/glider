import gym
import numpy as np
from scipy.integrate import solve_ivp


class Glider(gym.Env):
    def __init__(
        self,
        rho_s: float = 1,
        rho_f: float = 0.5,
        beta0: float = 0.1,
        u0: float = 1,
        v0: float = 1,
        w0: float = 0,
        x0: float = 0,
        y0: float = 0,
        theta0: float = 0,
        terminal_y: float = -10,
    ):
        """
        This is the main glider class.

        From here I will implement all of the methods and do the learning.
        """
        super(Glider, self).__init__()
        self.t = 0
        self.dt = 0.1  # I need to find a good value for this.
        self.rho_s = rho_s
        self.rho_f = rho_f
        self.CT = 1.2
        self.CR = np.pi
        self.A = 1.4
        self.B = 1.0
        self.mu = 0.2
        self.nu = 0.2
        self.g = 9.81
        self.u0 = u0
        self.v0 = v0
        self.w0 = w0
        self.x0 = x0
        self.y0 = y0
        self.beta0 = beta0
        self.theta0 = theta0
        self.beta = [beta0]  # nothing happens with this initial beta. For now.
        self.u = [u0]
        self.v = [v0]
        self.w = [w0]
        self.x = [x0]
        self.y = [y0]
        self.theta = [theta0]
        self.terminal_y = terminal_y
        self.t_hist = [np.array([0])]
        self.ang_limit = np.pi / 2
        self.t_max = 500 * self.dt
        # state is (x, y, v, theta)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1, -1, -1, -1]), high=np.array([1, 1, 1, 1])
        )  # normalized observation space. The exact way to bound the alues
        # is a guess
        self.max_speed = 10
        self.max_x = 20
        self.max_y = np.abs(
            self.terminal_y
        )  # should center the [terminal_y, 0] interval
        self.action_space = gym.spaces.Discrete(5)

    def M(self, w: float) -> float:
        """
        Compute the fluid torque on the body.

        Using eqn (2.23) in Paoletti and Mahadevan we compute the
        effective fluid torque on the solid body given the instantaneous
        value of the parameters.

        M = [(V/L) \mu_\tau + \nu_\tau |w|]*w

        In the RL follow on to the original paper they have M = 0.2*M. I need
        to figure out which one is correct.

        Parameters
        ----------
        w : float
            Angular velocity of cylinder.

        Returns
        -------
        M : float
            Value of the fluid torque given the current parameters.
        """
        M = (self.mu + self.nu * np.abs(w)) * w
        return M

    def G(self, u: float, v: float) -> float:
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

        Returns
        -------
        G : float
            Value of fluid force affecting the vertical motion.
        """
        G = (
            (1 / np.pi)
            * (self.A - self.B * (u**2 - v**2) / self.speed(u, v) ** 2)
            * self.speed(u, v)
            * v
        )
        return G

    def F(self, u: float, v: float) -> float:
        """
        Return the value of the fluid force affecting longitudinal motion.

        This function computes the F given by equation 2.21 in the
        Paoletti/Mahadevan paper:

            F = 1/pi(A - B*(u^2 - v^2)/(u^2 + v^2))*sqrt(u^2 + v^2)*u

        Parameters
        ----------
        u : float
            Speed along longitudinal axis of cylinder.
        v : float
            Speed along vertical axis of cylinder.

        Returns
        -------
        F : float
            Value of fluid force affecting the longitudinal motion.
        """
        F = (
            (1 / np.pi)
            * (self.A - self.B * (u**2 - v**2) / self.speed(u, v) ** 2)
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

    def gamma(self, u: float, v: float, w: float) -> float:
        """
        Compute function describing the circulation around the body.

        Use equation 2.20 in Paoletti/Mahadevan paper:

            gamma = 2/pi(-C_t*uv/sqrt(u^2 + v^2) + C_r*w)

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
        gamma = (2 / np.pi) * (-self.CT * u * v / self.speed(u, v) + self.CR * w)
        return gamma

    def moi(self) -> float:
        """
        Compute the non dimensional moment of inertia.

        The non dimensional moment of inertia is given in equation 2.13 of the
        Paoletti/Mahadevan paper:

            I = (rho_s*b)/(rho_f*a) = (rho_s/rho_f)*beta

        where beta = b/a.

        Parameters
        ----------

        Returns
        -------
        moi : float
            Non dimensional moment of inertia.
        """
        moi = (self.rho_s / self.rho_f) * self.beta[-1]
        return moi

    def hit_ground(self, t: float, s: np.ndarray) -> float:
        """
        Return the y distance from the terminal y.

        I want to end the episode at a given terminal y. In order to do that
        I will pass in an event to the solve_ivp function. The event function
        must evaluate to zero when the terminal condition is satisfied.

        Parameters
        ----------
        t : float
            Current time.
        s : np.ndarray (n,)
            Current state of the system. Since the state is
            s = (u, v, w, x, y, theta) we need to check y[4] to see what the
            current y axis position is.
        """
        dist = s[4] - self.terminal_y
        return dist

    def flipped_up(self, t: float, s: np.ndarray) -> float:
        """
        Check whether or not the cylinder has tipped too far up.
        """
        pass

    def turned_down(self, t: float, s: np.ndarray) -> float:
        """
        Check wheter or not the cylinder has turned too far down.
        """
        pass

    def rotate_too_fast(self, t: float, s: np.ndarray) -> float:
        """
        Check whether or not the cylinder is rotating too fast.
        """
        pass

    def dynamical_eqns(self, t: float, s: np.ndarray) -> np.ndarray:
        """
        Give the dynamics of the system at a given time.

        Input the time and the state and return the dynamics describing
        how the state will evolve over time. The state of the system is
        s = (u, v, w, x, y, \theta). The dynamical equations equations are
        given by equations 2.14-2.19 in the Paoletti/Mahadevan paper.

        Parameters
        ----------
        t : float
            Current time.
        s : np.ndarray
            State of the system.

        Returns
        -------
        ds_dt : np.ndarray
            Vector containing equations for evolution of state variables.
        """
        u_dot = (
            (self.moi() + 1) * s[1] * s[2]
            - self.gamma(u=s[0], v=s[1], w=s[2]) * s[1]
            - np.sin(s[5])
            - self.F(u=s[0], v=s[1])
        ) / (self.moi() + self.beta[-1] ** 2)
        v_dot = (
            -(self.moi() + self.beta[-1] ** 2) * s[0] * s[2]
            + self.gamma(u=s[0], v=s[1], w=s[2]) * s[0]
            - np.cos(s[5])
            - self.G(u=s[0], v=s[1])
        ) / (self.moi() + 1)
        w_dot = ((self.beta[-1] ** 2 - 1) * s[0] * s[1] - self.M(w=s[2])) / (
            0.25
            * (
                self.moi() * (1 + self.beta[-1] ** 2)
                + 0.5 * (1 - self.beta[-1] ** 2) ** 2
            )
        )
        x_dot = s[0] * np.cos(s[5]) - s[1] * np.sin(s[5])
        y_dot = s[0] * np.sin(s[5]) + s[1] * np.cos(s[5])
        theta_dot = s[2]
        s_dot = np.hstack((u_dot, v_dot, w_dot, x_dot, y_dot, theta_dot))
        return s_dot

    def update_state_history(self, solution_object) -> None:
        """
        Update the state history using the solution object found with solve_ivp

        I am assuming that the first element of the solution is the initial
        condition so I don't append that to the state. From the examples in
        the docs it seems to be the case that the first element in each
        solution array is just the initial conidtion so I think this is a fine
        approach.

        Parameters
        ----------
        solution_object : Bunch
            Bunch object defined in Scipy that holds the solution and times
            at which the solution is found. The solution is contained in the
            y property of the object.
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

    def forward(self, beta: float) -> None:
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
        beta : float
            Cylinder aspect ratio chosen by controller.

        Returns
        -------
        sol_object : bunch object with relevant fields listed below
            t : np.ndarry (n_points,)
                Time points where solution was found.
            y : np.ndarray (n, n_points)
                Value of state at the n_points
        """
        self.beta.append(beta)  # first add the new beta to the list.
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
            ],
            # events = self.hit_ground
        )
        self.update_state_history(solution_object=sol_object)
        self.t_hist.append(sol_object.t[1:])
        self.t += self.dt

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
            Current angle of the
        Returns
        -------
        out_of_bounds : bool
            True or false if the angle is outside of bounds or not.
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
        beta : float
            Aspect ratio corresponding to integer.
        """
        # lookup_dict = {0: 0.1, 1: 0.5, 2: 1, 3: 5, 4: 10}
        lookup_dict = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5}
        beta = lookup_dict[action]
        return beta

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
        beta = self.action_lookup(action)
        self.forward(beta=beta)
        speed = self.speed(u=self.u[-1], v=self.v[-1])
        angle = self.theta[-1]
        x_pos = self.x[-1]
        norm_x_pos = x_pos / self.max_x
        y_pos = self.y[-1]
        norm_y_pos = y_pos / self.max_y
        norm_angle = angle / self.ang_limit
        norm_speed = speed / self.max_speed
        # Maybe x and y can't be measured. I will play with this
        obs = np.array(
            [norm_x_pos, norm_y_pos, norm_speed, norm_angle], dtype=np.float32
        )
        angle_out_of_bounds = self.check_angle_bound(angle=angle)
        hit_ground = self.check_hit_ground(y=y_pos)
        if angle_out_of_bounds:
            # large penalty for flipping and end episode
            reward = -1000
            done = True
            # print("Flipped over!")
        elif hit_ground:
            # reward for making progress in x and end episode
            reward = 5 * self.x[-1]
            done = True
            # print("Hit ground!")
        elif self.t > self.t_max:
            # make sure agent doesn't just hover forever but reward forward
            # progress.
            reward = -10 + 5 * self.x[-1]
            done = True
            # print("Ran out of time")
        else:
            reward = 2 * self.x[-1]
            done = False
        info = {}  # for now I won't log any additional info
        return obs, reward, done, info

    def reset(self):
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
        self.t_hist = [np.array([0])]
        self.t = 0
        speed = self.speed(u=self.u[-1], v=self.v[-1])
        angle = self.theta[-1]
        x_pos = self.x[-1]
        norm_x_pos = x_pos / self.max_x
        y_pos = self.y[-1]
        norm_y_pos = y_pos / self.max_y
        norm_angle = angle / self.ang_limit
        norm_speed = speed / self.max_speed
        # Maybe x and y can't be measured. I will play with this
        obs = np.array(
            [norm_x_pos, norm_y_pos, norm_speed, norm_angle], dtype=np.float32
        )
        return obs

    def render(self):
        pass
