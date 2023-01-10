import numpy as np
import quadrotor

class solver4:
    def __init__(self, horizon_length = 1000):
        self.m = quadrotor.MASS
        self.r = quadrotor.LENGTH
        self.I = quadrotor.INERTIA
        self.g = quadrotor.GRAVITY
        self.dt = quadrotor.DELTA_T
        self.N = horizon_length
        # Part 1 Qs and Rs
        self.Q1 = np.array([[1e+06, 0., 0., 0., 0., 0.],
                           [0., 1e+03, 0., 0., 0., 0.],
                           [0., 0., 1e+06, 0., 0., 0.],
                           [0., 0., 0., 1e+03, 0., 0.],
                           [0., 0., 0., 0., 1e+06, 0.],
                           [0., 0., 0., 0., 0., 1e+03]])
        self.Q2 = np.array([[1e+01, 0., 0., 0., 0., 0.],
                            [0., 1e+00, 0., 0., 0., 0.],
                            [0., 0., 1e+01, 0., 0., 0.],
                            [0., 0., 0., 1e+00, 0., 0.],
                            [0., 0., 0., 0., 2e+02, 0.],
                            [0., 0., 0., 0., 0., 2e+00]])
        self.R1 = np.array([[1e-02, 0.],
                           [0., 1e-02]])
        self.R2 = np.array([[1e+01, 0.],
                            [0., 1e+01]])
        # Inital Guesses
        self.ustar = np.tile(np.array([[self.m * self.g / 2, self.m * self.g / 2]]).transpose(), (1, self.N))
        self.xstar = self.get_states(np.array([0., 0., 0., 0., 0., 0.]), self.ustar)
        self.prevCost = 0
        self.alpha = 1

    def get_states(self, x0, u):
        xstar = np.tile(np.array([[0., 0., 0., 0., 0., 0.]]).transpose(), (1, self.N + 1))
        xstar[:, 0] = x0
        for i, control in enumerate(u.transpose()):
            xstar[:, i + 1] = quadrotor.get_next_state(xstar[:, i], control)
        return xstar

    def compute_cost(self, z, u):
        J = [0] * (u.shape[1])
        for i, (state, control) in enumerate(zip(z.transpose(), u.transpose())):
            if i == 500:
                J[i] = (((state - np.array([3., 0., 3., 0., np.pi / 2, 0.])).reshape((6, 1)).transpose()) @ self.Q1 @ (state - np.array([3., 0., 3., 0., np.pi / 2, 0.])).reshape((6, 1))) + ((control - self.ustar[:, i]).reshape((2, 1)).transpose() @ self.R1 @ (control - self.ustar[:, i]).reshape((2, 1)))
            else:
                J[i] = (((state - np.array([0., 0., 0., 0., 0., 0.])).reshape((6, 1)).transpose()) @ self.Q2 @ (state - np.array([0., 0., 0., 0., 0., 0.])).reshape((6, 1))) + ((control - self.ustar[:, i]).reshape((2, 1)).transpose() @ self.R2 @ (control - self.ustar[:, i]).reshape((2, 1)))
        return max(J).item()

    def get_quadratic_approximation_cost(self, z, u):
        An, Bn, Qn, qn, Rn, rn = [0] * (u.shape[1]), [0] * (u.shape[1]), [0] * (u.shape[1]), [0] * (u.shape[1]), [0] * (u.shape[1]), [0] * (u.shape[1])
        for i, (state, control) in enumerate(zip(z.transpose(), u.transpose())):
            An[i], Bn[i] = self.get_linearization(state, control)
            if i == 500:
                Qn[i], qn[i], Rn[i], rn[i] = self.Q1, (self.Q1 @ (state - np.array([3., 0., 3., 0., np.pi / 2, 0.])).reshape((6, 1))), self.R1,  (self.R1 @ (control - self.ustar[:, i]).reshape((2, 1)))
            else:
                Qn[i], qn[i], Rn[i], rn[i] = self.Q2, (self.Q2 @ (state - np.array([0., 0., 0., 0., 0., 0.])).reshape((6, 1))), self.R2, (self.R2 @ (control - self.ustar[:, i]).reshape((2, 1)))
        return An, Bn, Qn, qn, Rn, rn

    def get_linearization(self, z, u):
        A = np.array([[1., self.dt, 0., 0., 0., 0.],
                      [0., 1., 0., 0., (-(u[0] + u[1]) * self.dt * np.cos(z[4])) / self.m, 0.],
                      [0., 0., 1., self.dt, 0., 0.],
                      [0., 0., 0., 1., (-(u[0] + u[1]) * self.dt * np.sin(z[4])) / self.m, 0.],
                      [0., 0., 0., 0., 1., self.dt],
                      [0., 0., 0., 0., 0., 1.]])
        B = np.array([[0., 0.],
                      [-(self.dt * np.sin(z[4])) / self.m, -(self.dt * np.sin(z[4])) / self.m],
                      [0., 0.],
                      [(self.dt * np.cos(z[4])) / self.m, (self.dt * np.cos(z[4])) / self.m],
                      [0., 0.],
                      [self.r * self.dt / self.I, -self.r * self.dt / self.I]])
        return A, B

    def solve_iLQR_trajectory(self, A, B, Q, q, R, r):
        K_gains, k_feedforward, Pn, pn, Pn[self.N], pn[self.N] = [0] * self.N, [0] * self.N, [0] * (self.N + 1), [0] * (self.N + 1), Q[-1], q[-1]
        for i, (An, Bn, Qn, qn, Rn, rn) in reversed(list(enumerate(zip(A, B, Q, q, R, r)))):
            K_gains[i] = -np.linalg.inv(Rn + (Bn.transpose() @ Pn[i + 1] @ Bn)) @ (Bn.transpose() @ Pn[i + 1] @ An)
            Pn[i] = Qn + (An.transpose() @ Pn[i + 1] @ An) + (An.transpose() @ Pn[i + 1] @ Bn @ K_gains[i])
            k_feedforward[i] = -np.linalg.inv(Rn + (Bn.transpose() @ Pn[i + 1] @ Bn)) @ ((Bn.transpose() @ pn[i + 1]) + rn)
            pn[i] = qn + (An.transpose() @ pn[i + 1]) + (An.transpose() @ Pn[i + 1] @ Bn @ k_feedforward[i])
        return K_gains, k_feedforward

    def line_search(self, J, K, k):
        if J > self.prevCost:
            self.alpha = self.alpha / 2
            if self.alpha < 0.01:
                return self.alpha
            for i, (state, control) in enumerate(zip(self.xstar.transpose(), self.ustar.transpose())):
                state, control = state.reshape((6, 1)), control.reshape((2, 1))
                self.ustar[:, i] = ((K[i] @ state) + (self.alpha * k[i]) + np.array([[self.m * self.g / 2], [self.m * self.g / 2]])).reshape((2, ))
                self.xstar[:, i + 1] = quadrotor.get_next_state(state.reshape((6, )), self.ustar[:, i].reshape((2, )))
            newCurrCost = self.compute_cost(self.xstar, self.ustar)
            return self.line_search(newCurrCost, K, k)
        else:
            alpha = self.alpha
            self.alpha = 1
            return alpha

    def vertical_orientation_controller(self, state, i):
        J = self.compute_cost(self.xstar, self.ustar)
        An, Bn, Qn, qn, Rn, rn = self.get_quadratic_approximation_cost(self.xstar, self.ustar)
        K, k = self.solve_iLQR_trajectory(An, Bn, Qn, qn, Rn, rn)
        alpha = self.line_search(J, K, k)  # local optimization
        self.prevCost = J
        return ((K[i] @ state).reshape((2, 1)) + (alpha * k[i]) + np.array([[self.m * self.g / 2], [self.m * self.g / 2]])).reshape((2, ))



