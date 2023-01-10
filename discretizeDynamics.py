import numpy as np
import quadrotor


class solver:
    def __init__(self, horizon_length = 1000):
        self.m = quadrotor.MASS
        self.r = quadrotor.LENGTH
        self.I = quadrotor.INERTIA
        self.g = quadrotor.GRAVITY
        self.dt = quadrotor.DELTA_T
        self.N = horizon_length

    def nominal_controller(self, state, i):
        return np.array([self.m * self.g / 2, self.m * self.g / 2])
