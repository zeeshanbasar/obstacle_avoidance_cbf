import numpy as np
from src import limits

g = 9.81

class Obstacle():

    def __init__(self, po_x=0.0, po_y=0.0, po_r=50.0, delta=0.0):

        self.po_x = po_x
        self.po_y = po_y
        self.delta = delta
        self.po_r = po_r + self.delta
        self.po_r_true = po_r

    def h(self, x):

        px, py, _, _ = x

        return (px - self.po_x)**2 + (py - self.po_y)**2 - (self.po_r)**2

    def h_dot(self, x):

        px, py, psi, V = x

        e = np.array([px, py]) - np.array([self.po_x, self.po_y])
        d = np.array([np.cos(psi), np.sin(psi)])

        return 2*V*np.dot(e,d)

    def constraint(self, x, u):

        px, py, psi, V = x

        d = np.array([np.cos(psi), np.sin(psi)])
        d_perp = np.array([-np.sin(psi), np.cos(psi)])
        e = np.array([px, py]) - np.array([self.po_x, self.po_y])

        A = np.array([2*np.dot(e,d), 2*g*np.dot(e,d_perp)])

        b = -(2*V**2 + (limits.K_0 + limits.K_1)*self.h_dot(x) + limits.K_0*limits.K_1*self.h(x))

        return A, b

    def psi_1(self, x):
        return self.h_dot(x) + limits.K_0*self.h(x)
