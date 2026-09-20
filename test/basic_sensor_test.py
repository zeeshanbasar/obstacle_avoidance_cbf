import numpy as np
import matplotlib.pyplot as plt
from src import vehicle, rk45, limits, guidance, helpers, filter, obstacle, estimator

g = 9.81

##### INIT #####
V_0 = limits.V_CRUISE
a_0 = 0.0
R0 = 10.0
phi_0 = np.atan((V_0**2)/(g*R0))
t_phi_0 = np.tan(phi_0)
psi_0 = np.deg2rad(0.0)
px_0, py_0 = 400.0, 0.0

x0 = np.array([px_0, py_0, psi_0, V_0])
u0 = np.array([a_0, t_phi_0])

psi_g = np.pi
V_g = V_0
xg = np.array([-400.0, 0.0, psi_g, V_g])

t0, tf = 0.0, 200.0
dt = 0.02

po_x, po_y, r_o = 0.0, 0.0, 50.0
obs = obstacle.Obstacle(po_x=po_x, po_y=po_y, po_r=r_o)

sol = filter.Solver(x_dim=4, u_dim=2, dt=dt, max_n_obs=1)
sol.set_obstacles(obstacles=[obs])

for b in [False, True]:

    t, traj, uncert_inputs, inputs, slacks, _, _ = rk45.simulate_with_filter(f=vehicle.f, 
                                                                    sol=sol,
                                                                    x0=x0, 
                                                                    xg=xg,
                                                                    u0=guidance.nominal,
                                                                    h=dt,
                                                                    T=tf,
                                                                    sens=estimator.Sensor(b))

    px = traj[:, 0]
    py = traj[:, 1]
    psi = traj[:, 2]
    V = traj[:, 3]

    a_nom = uncert_inputs[:,0]
    phi_nom = np.arctan(uncert_inputs[:,1])

    a = inputs[:,0]
    phi = np.arctan(inputs[:,1])

    h = []
    for i in range(len(traj)):
        h.append(obs.h(traj[i,:]))

    print(f"min h = {min(h):.6f}")
    print(f"min h at t = {t[np.argmin(h)]:.2f} s")
    print(f"clearance = {(np.sqrt(min(h) + r_o**2) - r_o):.2f} m")
    print(f"path length = {helpers.path_length(px, py):.2f} m")
    print(f"max |a| = {np.max(np.abs(a)):.2f} m/s^2")
    print(f"max |phi| = {np.max(np.abs(phi)):.2f} rads")



    fig0, ax0 = plt.subplots(1, 1)
    ax0.plot(px, py, label='traj')
    ax0.add_patch(plt.Circle((obs.po_x, obs.po_y), obs.po_r,
                            facecolor='0.85', edgecolor='k', zorder=0))
    ax0.plot(px[0], py[0], 'o', label='start')
    ax0.plot(xg[0], xg[1], '*', markersize=12, label='goal')
    ax0.set_aspect('equal')
    ax0.legend()
    ax0.grid()

    fig, (ax1, ax2) = plt.subplots(2,1)

    ax1.plot(t, psi, label='psi')
    ax1.legend()
    ax1.grid()

    ax2.plot(t, V, label='V')
    ax2.legend()
    ax2.grid()

    fig2, (ax21, ax22) = plt.subplots(2,1)

    ax21.plot(t, a, label='a')
    ax21.plot(t, a_nom, "--", label='a_nom')
    ax21.legend()
    ax21.grid()

    ax22.plot(t, phi, label='phi')
    ax22.plot(t, phi_nom, "--", label='phi_nom')
    ax22.legend()
    ax22.grid()

    fig3, ax3 = plt.subplots(1,1)
    ax3.plot(t, slacks, label='slack')
    ax3.legend()
    ax3.grid()

    fig4, ax4 = plt.subplots(1,1)
    ax4.plot(t, h, label='h')
    ax4.axhline(0.0, color='r', lw=1)
    ax4.set_yscale('symlog')
    ax4.legend()
    ax4.grid()

    plt.show()