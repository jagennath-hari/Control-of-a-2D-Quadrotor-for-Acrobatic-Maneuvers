import numpy as np
import quadrotor


class solver2:
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

    def compute_LQR(self, A, B):
        list_of_P, list_of_K = [0] * (self.N + 1), [0] * self.N
        for stage in range(self.N + 1, 0, -1):
            if stage != self.N + 1:
                list_of_K[stage - 1] = -np.linalg.inv((B.transpose() @ list_of_P[stage] @ B) + self.R) @ (B.transpose() @ list_of_P[stage] @ A)
                list_of_P[stage - 1] = self.Q + (A.transpose() @ list_of_P[stage] @ A) + (A.transpose() @ list_of_P[stage] @ B @ list_of_K[stage - 1])
            else:
                list_of_P[stage - 1] = self.Q
        return list_of_K


    def stay_still_controller(self, state, i):
        A, B = self.get_linearization(np.zeros((6, )), np.array([self.m * self.g / 2, self.m * self.g / 2]))
        K = self.compute_LQR(A, B)
        return (K[i] @ (state - np.zeros((6, )))) + np.array([self.m * self.g / 2, self.m * self.g / 2])
