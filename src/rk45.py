import numpy as np

def rk4(f, t, x, u, h):

    y = x.copy()

    k1 = h * f(t, y, u)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1, u)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2, u)
    k4 = h * f(t + h, y + k3, u)
    
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

def simulate(f, x0, u0, h, T=None, nSteps=None):
    t = 0.0
    x = x0
    traj = [x0.copy()]
    u = u0 #change this to the controller

    if nSteps is None:
        nSteps = round(T/h)
        
    for _ in range(nSteps):
        x = rk4(f, t, x, u, h)
        t += h
        traj.append(x.copy())

    return np.array(traj)

    
