import numpy as np

def rk4(f, t, x, u, h):

    y = x.copy()

    k1 = h * f(t, y, u)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1, u)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2, u)
    k4 = h * f(t + h, y + k3, u)
    
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

def simulate(f, x0, xg, u0, h, T=None, nSteps=None):
    t = 0.0
    x = x0
    traj = [x0.copy()]
    u = u0 #change this to the controller
    # u = u0(x0, xg)

    if (T is None) == (nSteps is None):
        raise ValueError("Give exactly one of T or nSteps.")
    if nSteps is None:
        nSteps = round(T / h)
        
    for _ in range(nSteps):
        x = rk4(f, t, x, u, h)
        # u = u0(x, xg)
        t += h
        traj.append(x.copy())

        if np.linalg.norm(x[:2]-xg[:2]) < 0.01:
            break

    return np.array(traj)

    
