import json
import numpy as np
import matplotlib.pyplot as plt
from src import vehicle, rk45, limits, guidance, helpers, filter, obstacle, scenario_gen, estimator

SENSOR_SEED = 42
SCENE_SEED = 1337

g = 9.81
R_MIN_CRUISE = limits.V_CRUISE**2 / (g * np.tan(limits.PHI_MAX))   # 33.1 m
R_BLEND = 4.0 * R_MIN_CRUISE
scene = scenario_gen.Scenario(SCENE_SEED)
t0, tf = 0.0, 200.0
dt = 0.02
n_obs = 2
delta = 0.0

def corridor(gap, r_o=50.0, delta=0.0, entry_offset=0.0, entry_heading=0.0,
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
       obstacle.Obstacle(po_x=+0.5*stagger, po_y=-c, po_r=r_o, delta=delta)]

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

sens = estimator.Sensor()

sens.reseed(SENSOR_SEED)
scene.reseed(SCENE_SEED)

##### INIT #####
x0 = scene.initial_condition()
R_min = x0[3]**2 / (g * np.tan(limits.PHI_MAX))

isFinished = False

for gap in [40,20,10]:
    output = {}
    for stagger in [70, 95, 120]:

        x0, xg, obs = corridor(gap, delta=delta, stagger=stagger)
        a_0, t_phi_0 = scene.initial_control(x=x0,goal=xg)

        sol.set_obstacles(obstacles=obs)

        sol.reset(tphi0=t_phi_0)

        sens.reseed(SENSOR_SEED)

        assert sens.t_next == 0.0
        assert sens.x_hat is None

        output[stagger] = {
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
            'margin_breach': False
        }

        # if np.linalg.norm(x0[:2] - np.array([po_x, po_y])) <= obs.po_r_true:
        #     output['scenario_condition'] = 'fail; init inside obstacle'
        #     continue
        
        # if np.linalg.norm(xg[:2] - np.array([po_x, po_y])) <= obs.po_r_true:
        #     output['scenario_condition'] = 'fail; term inside obstacle'
        #     continue

        # if np.linalg.norm(np.array([po_x, po_y]) - x0[:2]) < obs.po_r_true + R_min:
        #     output['scenario_condition'] = 'fail; obstacle inside minimum turn radius'
        #     continue

        t, traj, uncert_inputs, inputs, slacks,_,_ = rk45.simulate_with_filter(f=vehicle.f, 
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


        d_min = np.array([np.min(np.linalg.norm(traj[:, :2] - np.array([o.po_x, o.po_y]), axis=1)) for o in obs])
        
        output[stagger]['min |h|'] = np.min(np.array(h), axis=0).tolist()
        output[stagger]['clearance'] = [d_min[i] - obs[i].po_r_true for i in range(len(obs))]
        output[stagger]['path length'] = helpers.path_length(px, py)
        output[stagger]['max |a|'] = np.max(np.abs(a))
        output[stagger]['max |phi|'] = np.max(np.abs(phi))
        output[stagger]['max_da'] = float(np.max(np.abs(a - a_nom)))
        output[stagger]['max_dphi'] = float(np.max(np.abs(phi - phi_nom)))
        output[stagger]['slack'] = slacks.tolist()
        output[stagger]['violation'] = [bool(d_min[i] < obs[i].po_r_true) for i in range(len(obs))]
        output[stagger]['guarantee_void'] = [bool(np.max(slacks, axis=0)[i] > 1e-6) for i in range(len(obs))]
        output[stagger]['margin_breach'] = [bool(d_min[i] < obs[i].po_r_true + obs[i].delta) for i in range(len(obs))]

        if np.linalg.norm(traj[-1, :2]-xg[:2]) > 5.0:
            output[stagger]['scenario_condition'] = f'terminal condition not reached in {tf} s'
        else:
            output[stagger]['scenario_condition'] = 'pass'

    with open(f'multiObs_corridor_gap_{gap}_stagger_perfectSense.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)


# fig0, ax0 = plt.subplots(1, 1)
# ax0.plot(px, py, label='traj')
# for o in obs:
#     ax0.add_patch(plt.Circle((o.po_x, o.po_y), o.po_r,
#                             facecolor='0.85', edgecolor='k', zorder=0))
# ax0.plot(px[0], py[0], 'o', label='start')
# ax0.plot(xg[0], xg[1], '*', markersize=12, label='goal')
# ax0.set_aspect('equal')
# ax0.legend()
# ax0.grid()

# plt.show()

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