import numpy as np
import quadrotor


class solver3:
    def __init__(self, horizon_length = 1000):
        self.m = quadrotor.MASS
        self.r = quadrotor.LENGTH
        self.I = quadrotor.INERTIA
        self.g = quadrotor.GRAVITY
        self.dt = quadrotor.DELTA_T
        self.N = horizon_length
        self.Q = np.array([[1e+200, 0., 0., 0., 0., 0.],
                           [0., 1e+20, 0., 0., 0., 0.],
                           [0., 0., 1e+200, 0., 0., 0.],
                           [0., 0., 0., 1e+20, 0., 0.],
                           [0., 0., 0., 0., 1e+200, 0.],
                           [0., 0., 0., 0., 0., 1e+20]])
        self.R = np.array([[1e-05, 0.],
                           [0., 1e-05]])

        self.thetas = np.linspace(np.pi / 2, 5 * np.pi / 2, self.N + 1)
        self.move_in_a_circle = np.zeros((quadrotor.NUMBER_STATES, self.N + 1))
        for i, theta in enumerate(self.thetas):
            self.move_in_a_circle[:, i] = np.array([np.sin(theta),
                                                    0.,
                                                    np.cos(theta),
                                                    0.,
                                                    0.,
                                                    0.])
        self.move_in_a_circle_tilted = np.zeros((quadrotor.NUMBER_STATES, self.N + 1))
        for i, theta in enumerate(self.thetas):
            self.move_in_a_circle_tilted[:, i] = np.array([np.sin(theta),
                                                           0.,
                                                           np.cos(theta),
                                                           0.,
                                                           np.pi / 4,
                                                           0.])

        self.ustar = np.array([self.m * self.g / 2, self.m * self.g / 2])

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

    def solve_LQR_trajectory(self, A, B, x_bar):
        K_gains, k_feedforward, Pn, pn, Pn[self.N], pn[self.N] = [0] * self.N, [0] * self.N, [0] * (self.N + 1), [0] * (self.N + 1), self.Q, -self.Q @ x_bar[:, self.N]
        for i in range(self.N - 1, -1, -1):
            K_gains[i] = -np.linalg.inv(self.R + (B.transpose() @ Pn[i + 1] @ B)) @ (B.transpose() @ Pn[i + 1] @ A)
            Pn[i] = self.Q + (A.transpose() @ Pn[i + 1] @ A) + (A.transpose() @ Pn[i + 1] @ B @ K_gains[i])
            k_feedforward[i] = -np.linalg.inv(self.R + (B.transpose() @ Pn[i + 1] @ B)) @ (B.transpose() @ pn[i + 1])
            pn[i] = (-self.Q @ x_bar[:, i]) + (A.transpose() @ pn[i + 1]) + (A.transpose() @ Pn[i + 1] @ B @ k_feedforward[i])
        return K_gains, k_feedforward

    def move_circular_controller(self, state, i):
        A, B = self.get_linearization(self.move_in_a_circle[:, i], np.array([self.m * self.g / 2, self.m * self.g / 2]))
        K, k = self.solve_LQR_trajectory(A, B, self.move_in_a_circle)
        return (K[i] @ state) + k[i] + np.array([self.m * self.g / 2, self.m * self.g / 2])

    def move_circular_tilted_controller(self, state, i):
        A, B = self.get_linearization(self.move_in_a_circle_tilted[:, i], np.array([self.m * self.g / 2, self.m * self.g / 2]))
        K, k = self.solve_LQR_trajectory(A, B, self.move_in_a_circle_tilted)
        return (K[i] @ state) + k[i] + np.array([self.m * self.g / 2, self.m * self.g / 2])
