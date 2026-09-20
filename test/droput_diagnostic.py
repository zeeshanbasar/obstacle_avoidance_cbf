import json
import numpy as np
import matplotlib.pyplot as plt
from src import vehicle, rk45, limits, guidance, helpers, filter, obstacle, scenario_gen, estimator

SENSOR_SEED = 42
SCENE_SEED = 1337
DROP_SEED = 0

g = 9.81
R_MIN_CRUISE = limits.V_CRUISE**2 / (g * np.tan(limits.PHI_MAX))   # 33.1 m
R_BLEND = 4.0 * R_MIN_CRUISE
scene = scenario_gen.Scenario(SCENE_SEED)
t0, tf = 0.0, 200.0
dt = 0.02
n_obs = 2
delta = 2.0
gap = 10
stagger = 120
jscenario = 497
output = {}

def corridor(gap, r_o=50.0, delta=0.0, entry_offset=50.0, entry_heading=0.0,
             stagger=0.0,
             V_0=limits.V_CRUISE, L_in=200.0, L_out=400.0):

    c = 0.5 * gap + r_o

    if stagger==0.0:
        assert gap - 2.0 * delta > 0.0
        assert abs(entry_offset) < 0.5 * gap
    else:
        assert stagger**2 + (2.0*c)**2 > (2.0*(r_o + delta))**2

    assert L_in > R_BLEND

    obs = [obstacle.Obstacle(po_x=-0.5*stagger, po_y=+c, po_r=r_o, delta=delta),
    #    obstacle.Obstacle(po_x=+0.5*stagger, po_y=-c, po_r=r_o, delta=delta)
       ]

    x0 = np.array([-L_in, entry_offset, entry_heading, V_0])
    xg = np.array([+L_out, 0.0, 0.0, V_0])
    return x0, xg, obs

def corridor_gap(x0,xg,obs):

    p0 = np.asarray(x0[:2], dtype=float)
    p1 = np.asarray(xg[:2], dtype=float)
    c = np.array([obs.po_x, obs.po_y])

    v = p1 - p0
    L2 = np.dot(v, v)
    t = 0.0 if L2 == 0.0 else np.clip(np.dot(c - p0, v) / L2, 0.0, 1.0)
    closest = p0 + t * v

    return np.linalg.norm(c - closest) - obs.po_r_true

output = {}

# Built once. The obstacle enters only through sol.obs, so it is swapped per
# scenario instead of reconstructing the QP 500 times.
sol = filter.Solver(x_dim=4, u_dim=2, obs=None, dt=dt, max_n_obs=2)


scene.reseed(SCENE_SEED + jscenario)

##### INIT #####
po_x, po_y, r_o = scene.obstacle()
x0 = scene.initial_condition()
xg = scene.terminal_condition()
obs = [obstacle.Obstacle(po_x=po_x, po_y=po_y, po_r=r_o, delta=delta)]
R_min = x0[3]**2 / (g * np.tan(limits.PHI_MAX))
a_0, t_phi_0 = scene.initial_control(x=x0,goal=xg)

sol.set_obstacles(obstacles=obs)

fig0, ax0 = plt.subplots(1, 1)
for o in obs:
    ax0.add_patch(plt.Circle((o.po_x, o.po_y), o.po_r,
                            facecolor='0.85', edgecolor='k', zorder=0))

for drop in [True, False]:

    sol.reset(tphi0=t_phi_0)

    sens = estimator.Sensor(isDrop=drop, p_drop=0.5, rate_hz=5)

    sens.reseed(SENSOR_SEED + jscenario, DROP_SEED + jscenario)

    assert sens.t_next == 0.0
    assert sens.x_hat is None

    output[drop] = {
        'scenario_condition': None,
        'initial_condition': x0.tolist(),
        'terminal condition': xg.tolist(),
        'obstacle_condition': [[o.po_x, o.po_y, o.po_r_true] for o in obs],
        'gap_norm': [corridor_gap(x0,xg,o)/o.po_r_true for o in obs],
        # 'offset_norm': [scene.offset/o.po_r_true for o in obs],
        'min |h|': 0,
        'clearance': 0,
        'path length': 0,
        'max |a|': 0,
        'max |phi|': 0,
        'max_da': 0,
        'max_dphi': 0,
        'max slack': 0,
        'violation': False,
        'guarantee_void': False,
        'margin_breach': False,
        'traj': [],
        'x_hat': [],
        'age':[]
    }


    t, traj, uncert_inputs, inputs, slacks, x_hat, age = rk45.simulate_with_filter(f=vehicle.f, 
                                                                    sol=sol,
                                                                    x0=x0, 
                                                                    xg=xg,
                                                                    u0=guidance.nominal,
                                                                    h=dt,
                                                                    T=tf,
                                                                    sens=sens)

    px = traj[:, 0]
    py = traj[:, 1]
    psi = traj[:, 2]
    V = traj[:, 3]

    a_nom = uncert_inputs[:,0]
    phi_nom = np.arctan(uncert_inputs[:,1])

    a = inputs[:,0]
    phi = np.arctan(inputs[:,1])

    h = []
    for j in range(len(traj)):
        h.append([o.h(traj[j,:]) for o in obs])

    e = x_hat[:, :2] - traj[:, :2]
    along = e[:, 0]*np.cos(psi) + e[:, 1]*np.sin(psi)
    cross = -e[:, 0]*np.sin(psi) + e[:, 1]*np.cos(psi)
    k = int(np.argmax(age))
    print(f'max age {age[k]:.2f} s: along {along[k]:+.1f} m, cross {cross[k]:+.1f} m')
    print(f'over the run: |along| max {np.abs(along).max():.1f} m, |cross| max {np.abs(cross).max():.1f} m')
    print(f'  V={V[k]:.1f} m/s, h={h[k][0]:.0f}, V*age={V[k]*age[k]:.1f} m')


    d_min = np.array([np.min(np.linalg.norm(traj[:, :2] - np.array([o.po_x, o.po_y]), axis=1)) for o in obs])

    output[drop]['min |h|'] = np.min(np.array(h), axis=0).tolist()
    output[drop]['clearance'] = [d_min[i] - obs[i].po_r_true for i in range(len(obs))]
    output[drop]['path length'] = helpers.path_length(px, py)
    output[drop]['max |a|'] = np.max(np.abs(a))
    output[drop]['max |phi|'] = np.max(np.abs(phi))
    output[drop]['max_da'] = float(np.max(np.abs(a - a_nom)))
    output[drop]['max_dphi'] = float(np.max(np.abs(phi - phi_nom)))
    output[drop]['slack'] = slacks.tolist()
    output[drop]['violation'] = [bool(d_min[i] < obs[i].po_r_true) for i in range(len(obs))]
    output[drop]['guarantee_void'] = [bool(np.max(slacks, axis=0)[i] > 1e-6) for i in range(len(obs))]
    output[drop]['margin_breach'] = [bool(d_min[i] < obs[i].po_r_true + obs[i].delta) for i in range(len(obs))]
    output[drop]['traj'] = traj.tolist()
    output[drop]['x_hat'] = x_hat.tolist()
    output[drop]['age'] = age.tolist()


    if np.linalg.norm(traj[-1, :2]-xg[:2]) > 5.0:
        output[drop]['scenario_condition'] = f'terminal condition not reached in {tf} s'
    else:
        output[drop]['scenario_condition'] = 'pass'

    ax0.plot(px, py)

with open(f'corridor_gap_{gap}_stagger_perfectSense_dropoutTest_{jscenario}.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)







ax0.legend(["Drop True", "Drop False"])
ax0.plot(px[0], py[0], 'o', label='start')
ax0.plot(xg[0], xg[1], '*', markersize=12, label='goal')
ax0.set_aspect('equal')
ax0.legend()
ax0.grid()

plt.show()

# fig, (ax1, ax2) = plt.subplots(2,1)

# ax1.plot(t, psi, label='psi')
# ax1.legend()
# ax1.grid()

# ax2.plot(t, V, label='V')
# ax2.legend()
# ax2.grid()

# fig2, (ax21, ax22) = plt.subplots(2,1)

# ax21.plot(t, a, label='a')
# ax21.plot(t, a_nom, "--", label='a_nom')
# ax21.legend()
# ax21.grid()

# ax22.plot(t, phi, label='phi')
# ax22.plot(t, phi_nom, "--", label='phi_nom')
# ax22.legend()
# ax22.grid()

# fig3, ax3 = plt.subplots(1,1)
# ax3.plot(t, slacks, label='slack')
# ax3.legend()
# ax3.grid()

# fig4, ax4 = plt.subplots(1,1)
# ax4.plot(t, h, label='h')
# ax4.axhline(0.0, color='r', lw=1)
# ax4.set_yscale('symlog')
# ax4.legend()
# ax4.grid()

# plt.show()