import numpy as np
import matplotlib.pyplot as plt
from src import vehicle, rk45, limits

g = 9.81

##### INIT #####
V_0 = limits.V_CRUISE
a_0 = 0.0
R0 = 10.0
phi_0 = np.atan((V_0**2)/(g*R0))
t_phi_0 = np.tan(phi_0)
psi_0 = np.deg2rad(0.0)
px_0, py_0 = 0.0, 0.0

x0 = np.array([px_0, py_0, psi_0, V_0])
u0 = np.array([a_0, t_phi_0])

##### basic check #####
t0, tf = 0.0, 1.0
h = 0.02

traj = rk45.simulate(f=vehicle.f, 
                     x0=x0, 
                     xg=np.array([0.0,0.0,0.0,0.0]),
                     u0=u0,
                     h=h,
                     T=tf)

px = traj[:, 0]
py = traj[:, 1]

plt.plot(px, py)
plt.grid()
plt.show()