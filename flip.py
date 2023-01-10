import numpy as np
import quadrotor


class quadSys():
    def __init__(self):
        self.mass = quadrotor.MASS
        self.length = quadrotor.LENGTH
        self.inertia = quadrotor.INERTIA
        self.gravity = quadrotor.GRAVITY
        self.delta = quadrotor.DELTA_T
        self.numS = quadrotor.NUMBER_STATES
        self.numU = quadrotor.NUMBER_CONTROLS
        self.horizon = 1000
        self.Kn = None
        self.z1 = np.zeros((6, 1))
        self.z2 = np.matrix([1.5, 0, 3, 0, np.pi, 0]).T
        self.z3 = np.matrix([3, 0, 0, 0, 2 * np.pi, 0]).T
        self.u = np.array([[self.mass * self.gravity / 2], [self.mass * self.gravity / 2]])

        self.prevCost = 1e+20

        # Initial Guess
        self.uS = np.tile(np.array([[self.mass * self.gravity / 2, self.mass * self.gravity / 2]]).transpose(),
                          (1, self.horizon))
        self.zS = self.get_states(np.array([0., 0., 0., 0., 0., 0.]), self.uS)
        self.Q = np.array([[5, 0., 0., 0., 0., 0.],
                           [0., 2, 0., 0., 0., 0.],
                           [0., 0., 5, 0., 0., 0.],
                           [0., 0., 0., 2, 0., 0.],
                           [0., 0., 0., 0., 2, 0.],
                           [0., 0., 0., 0., 0., 2]])
        self.Q1 = np.array([[10, 0., 0., 0., 0., 0.],
                            [0., 2, 0., 0., 0., 0.],
                            [0., 0., 100, 0., 0., 0.],
                            [0., 0., 0., 2, 0., 0.],
                            [0., 0., 0., 0., 200, 0.],
                            [0., 0., 0., 0., 0., 2]])
        self.Q2 = np.array([[15.5, 0., 0., 0., 0., 0.],
                            [0., 2, 0., 0., 0., 0.],
                            [0., 0., 20.5, 0., 0., 0.],
                            [0., 0., 0., 2, 0., 0.],
                            [0., 0., 0., 0., 200, 0.],
                            [0., 0., 0., 0., 0., 2]])
        self.R = np.array([[1, 0.],
                           [0., 1]])

    def get_states(self, x0, u):
        xstar = np.tile(np.array([[0., 0., 0., 0., 0., 0.]]).transpose(), (1, self.horizon + 1))
        xstar[:, 0] = x0
        for i, control in enumerate(u.transpose()):
            xstar[:, i + 1] = quadrotor.get_next_state(xstar[:, i], control)
        return xstar

    def get_next_state(self, z, u):

        x = z[0, 0]
        vx = z[1, 0]
        y = z[2, 0]
        vy = z[3, 0]
        theta = z[4, 0]
        omega = z[5, 0]

        dydt = np.zeros([quadrotor.NUMBER_STATES, 1])
        dydt[0, 0] = vx
        dydt[1, 0] = (-(u[0, 0] + u[1, 0]) * np.sin(theta)) / quadrotor.MASS
        dydt[2, 0] = vy
        dydt[3, 0] = ((u[0, 0] + u[1, 0]) * np.cos(theta) - quadrotor.MASS * quadrotor.GRAVITY) / quadrotor.MASS
        dydt[4, 0] = omega
        dydt[5, 0] = (quadrotor.LENGTH * (u[0, 0] - u[1, 0])) / quadrotor.INERTIA
        z_next = z + dydt * quadrotor.DELTA_T

        return z_next

    def get_linearization(self, z, u):
        A = np.matrix([[1, self.delta, 0, 0, 0, 0],
                       [0, 1, 0, 0, -(1 / self.mass) * (u[0, 0] + u[1, 0]) * np.cos(z[4, 0]) * self.delta, 0],
                       [0, 0, 1, self.delta, 0, 0],
                       [0, 0, 0, 1, -(1 / self.mass) * (u[0, 0] + u[1, 0]) * np.sin(z[4, 0]) * self.delta, 0],
                       [0, 0, 0, 0, 1, self.delta], [0, 0, 0, 0, 0, 1]])
        B = np.matrix(
            [[0, 0], [-np.sin(z[4, 0]) * self.delta / self.mass, -np.sin(z[4, 0]) * self.delta / self.mass], [0, 0],
             [np.cos(z[4, 0]) * self.delta / self.mass, np.cos(z[4, 0]) * self.delta / self.mass], [0, 0],
             [self.length * self.delta / self.inertia, -self.length * self.delta / self.inertia]])
        return A, B

    def computeCost(self, z, u, horizonLength):
        cost = 0
        for i in range(horizonLength - 1):
            if 450 <= i <= 550:
                cost += (z[:, [i]] - self.z2).T @ self.Q1 @ (z[:, [i]] - self.z2) + (u[:, [i]] - self.u).T @ self.R @ (
                        u[:, [i]] - self.u)
            elif i < 450:
                cost += (z[:, [i]] - self.z1).T @ self.Q @ (z[:, [i]] - self.z1) + (u[:, [i]] - self.u).T @ self.R @ (
                        u[:, [i]] - self.u)
                cost += (z[:, [horizonLength - 1]] - self.z1).T @ self.Q @ (z[:, [horizonLength - 1]] - self.z1)
            else:
                cost += (z[:, [i]] - self.z3).T @ self.Q2 @ (z[:, [i]] - self.z3) + (u[:, [i]] - self.u).T @ self.R @ (
                        u[:, [i]] - self.u)
                cost += (z[:, [horizonLength - 1]] - self.z1).T @ self.Q @ (z[:, [horizonLength - 1]] - self.z1)
        return cost

    def getQuadraticApproximation(self, zS, uS, horizonLength):
        listOfA = []
        listOfB = []
        listOfq = []
        listOfr = []

        for i in range(horizonLength - 1):

            a, b = self.get_linearization(zS[:, [i]], uS[:, [i]])
            listOfA.append(a)
            listOfB.append(b)
            if 450 <= i <= 550:
                q, r = self.Q1 @ (zS[:, [i]] - self.z2), self.R @ (uS[:, [i]] - self.u)
            elif i < 450:
                q, r = self.Q @ (zS[:, [i]] - self.z1), self.R @ (uS[:, [i]] - self.u)
            else:
                q, r = self.Q2 @ (zS[:, [i]] - self.z3), self.R @ (uS[:, [i]] - self.u)

            listOfq.append(q)
            listOfr.append(r)
        q = self.Q @ (zS[:, [-1]] - self.z1)
        listOfq.append(q)
        return listOfA, listOfB, listOfq, listOfr

    def lineSearch(self, K_gains, k_feedforward):
        alpha = 1
        while alpha > 0.01:
            z0 = np.zeros((6, 1))
            for i in range(self.horizon - 1):
                if i != 0:
                    z = np.column_stack((z, zTemp))
                    uTemp = self.uS[:, [i]] + K_gains[i] @ (zTemp - self.zS[:, [i]]) + alpha * k_feedforward[i]
                    u = np.column_stack((u, uTemp))
                    zTemp = self.get_next_state(zTemp, uTemp)
                else:
                    zTemp = np.zeros((6, 1))
                    uTemp = self.uS[:, [i]] + K_gains[i] @ (zTemp - self.zS[:, [i]]) + alpha * k_feedforward[i]
                    z = zTemp
                    u = uTemp
                    zTemp = self.get_next_state(zTemp, uTemp)
                    zTemp = zTemp.reshape([6, 1])
            z = np.column_stack((z, zTemp))
            cost = self.computeCost(z, u, horizonLength=1000)

            if cost < self.prevCost:
                self.prevCost = cost
                self.zS = z
                self.uS = u
                return True
            else:

                alpha *= 0.5
        return False

    def full_flip_controller(self, state, i):
        return self.uS[:, [i]].reshape([2, ])

    def ilrq(self):
        n = 0
        while True:
            n = n + 1
            K_gains = []
            k_feedforward = []
            listOfA, listOfB, listOfq, listOfr = self.getQuadraticApproximation(self.zS, self.uS, self.horizon)
            Pn = self.Q2

            pn = listOfq[-1]
            for i in range(self.horizon - 1):
                q = listOfq[self.horizon - 2 - i]
                A = listOfA[self.horizon - 2 - i]
                B = listOfB[self.horizon - 2 - i]
                r = listOfr[self.horizon - 2 - i]
                if 450 <= i <= 550:
                    Kn = -np.linalg.inv(self.R + B.transpose() @ Pn @ B) @ B.transpose() @ Pn @ A
                    P = self.Q1 + A.transpose() @ Pn @ A + A.transpose() @ Pn @ B @ Kn
                    kn = -np.linalg.inv(self.R + B.transpose() @ Pn @ B) @ (B.transpose() @ pn + r)
                    p = q + A.transpose() @ pn + A.transpose() @ Pn @ B @ kn
                elif i < 450:
                    Kn = -np.linalg.inv(self.R + B.transpose() @ Pn @ B) @ B.transpose() @ Pn @ A
                    P = self.Q2 + A.transpose() @ Pn @ A + A.transpose() @ Pn @ B @ Kn
                    kn = -np.linalg.inv(self.R + B.transpose() @ Pn @ B) @ (B.transpose() @ pn + r)
                    p = q + A.transpose() @ pn + A.transpose() @ Pn @ B @ kn
                else:
                    Kn = -np.linalg.inv(self.R + B.transpose() @ Pn @ B) @ B.transpose() @ Pn @ A
                    P = self.Q + A.transpose() @ Pn @ A + A.transpose() @ Pn @ B @ Kn
                    kn = -np.linalg.inv(self.R + B.transpose() @ Pn @ B) @ (B.transpose() @ pn + r)
                    p = q + A.transpose() @ pn + A.transpose() @ Pn @ B @ kn

                Pn = P
                pn = p

                K_gains.append(Kn)
                k_feedforward.append(kn[:, 0])

            K_gains = K_gains[::-1]
            k_feedforward = k_feedforward[::-1]

            if self.lineSearch(K_gains, k_feedforward):
                pass
            else:
                break
            if n == 15:
                break
