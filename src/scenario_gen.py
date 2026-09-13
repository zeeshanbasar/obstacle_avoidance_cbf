"""
randomise:
    - obs center
    - obs radius
    - start pos
    - start heading
    - start speed
    - lateral offset of the goal from the start-obstacle ray
fix:
    - seed

add to limits:
    X_MAX = Y_MAX = 1000.0
    XO_MAX = YO_MAX = 850.0
    RO_MAX = 100.0
    RO_MIN = 10.0

Call order MUST be: obstacle() -> initial_condition() -> terminal_condition()
-> initial_control(). Later methods read state set by earlier ones.
"""

import numpy as np
from src import limits, guidance

SEED = 1337


G = 9.81

class Scenario:

    def __init__(self, SCENE_SEED):
        self.rng = np.random.default_rng(SCENE_SEED)

    def reseed(self, SCENE_SEED):
        self.rng = np.random.default_rng(SCENE_SEED)

    def obstacle(self):

        self.po_x = self.rng.uniform(-limits.XO_MAX, limits.XO_MAX)
        self.po_y = self.rng.uniform(-limits.YO_MAX, limits.YO_MAX)
        self.po_r = self.rng.uniform(limits.RO_MIN, limits.RO_MAX)

        self.o = np.array([self.po_x, self.po_y, self.po_r])

        return self.po_x, self.po_y, self.po_r

    def initial_condition(self):

        self.px_0 = self.rng.uniform(-limits.X_MAX, limits.X_MAX)
        self.py_0 = self.rng.uniform(-limits.Y_MAX, limits.Y_MAX)
        self.psi_0 = self.rng.uniform(-np.pi, np.pi)
        self.V_0 = self.rng.uniform(limits.V_STALL, limits.V_MAX)

        self.x0 = np.array([self.px_0, self.py_0, self.psi_0, self.V_0])

        return self.x0

    def initial_control(self, x=None, goal=None):
        self.a_0, self.tan_phi_0 = guidance.nominal(x=np.array([self.px_0, self.py_0, self.psi_0, self.V_0] if x is None else x),
                                                    goal=np.array([self.px_g, self.py_g, self.psi_g, self.V_g] if goal is None else goal))

        return np.array([self.a_0, self.tan_phi_0])

    def terminal_condition(self):

        R_min = self.x0[3]**2 / (G * np.tan(limits.PHI_MAX))

        d = 2*(self.po_r + R_min) + self.rng.uniform(0.5*R_min, 5*R_min)

        # Unit vector from start to obstacle, and its left normal.
        u = self.o[:2] - self.x0[:2]
        u = u / np.linalg.norm(u)
        n = np.array([-u[1], u[0]])

        # Lateral offset of the goal from the start-obstacle ray, in metres.
        # offset = 0 is the degenerate head-on case; |offset| = po_r grazes.
        self.offset = self.rng.uniform(-1.0, 1.0) * self.po_r

        p_g = self.o[:2] + d*u + self.offset*n

        self.px_g, self.py_g = float(p_g[0]), float(p_g[1])
        self.psi_g = float(np.arctan2(u[1], u[0]))
        self.V_g = self.rng.uniform(limits.V_STALL, limits.V_MAX)

        return np.array([self.px_g, self.py_g, self.psi_g, self.V_g])