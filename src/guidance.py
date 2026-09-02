import numpy as np
from src import helpers, limits

g = 9.81

def nominal(x, goal):

    px, py, psi, V = x
    px_g, py_g, psi_g, V_g = goal

    range = np.hypot(px_g - px, py_g - py)
    R_min_cruise = (limits.V_CRUISE**2)/(g*np.tan(limits.PHI_MAX))
    R_blend = 4*R_min_cruise

    w = np.clip(range/R_blend, 0, 1)

    psi_bearing = np.atan2((py_g - py), (px_g - px))
    psi_ref = helpers.heading_wrap(psi_g + w*helpers.heading_wrap(psi_bearing - psi_g))


    K_PSI = 1.0
    phi_c = np.clip(K_PSI*helpers.heading_wrap(psi_ref - psi), -limits.PHI_MAX, limits.PHI_MAX)
    t_phi_c = np.tan(phi_c)

    K_V = 0.5
    a = np.clip(K_V*(limits.V_CRUISE - V), -limits.A_MAX, limits.A_MAX)

    u = np.array([a, t_phi_c])

    return u