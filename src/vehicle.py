import numpy as np

g = 9.81

def f(t, x, u):

    px, py, psi, V = x
    a, t_phi = u

    dx = np.array([[V*np.cos(psi)], 
                   [V*np.sin(psi)],
                   [0.0],
                   [0.0]]) + np.array([[0.0, 0.0],
                                       [0.0, 0.0],
                                       [0.0, g/V],
                                       [1.0, 0.0]]) @ np.array([[a],
                                                                [t_phi]])

    return dx.reshape(np.size(x))