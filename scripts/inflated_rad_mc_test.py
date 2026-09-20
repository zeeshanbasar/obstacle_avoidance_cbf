import json
import numpy as np
import matplotlib.pyplot as plt
from src import vehicle, rk45, limits, guidance, helpers, filter, obstacle, scenario_gen, estimator

SCENE_SEED = 1337
SENSE_SEED = 42

g = 9.81
scene = scenario_gen.Scenario(SCENE_SEED)
t0, tf = 0.0, 200.0
dt = 0.02

output = {}

# Built once. The obstacle enters only through sol.obs, so it is swapped per
# scenario instead of reconstructing the QP 500 times.
sol = filter.Solver(x_dim=4, u_dim=2, obs=None, dt=dt, max_n_obs=1)
sens = estimator.Sensor()

def corridor_gap(x0,xg,obs):

    p0 = np.asarray(x0[:2], dtype=float)
    p1 = np.asarray(xg[:2], dtype=float)
    c = np.array([obs[0], obs[1]])

    v = p1 - p0
    L2 = np.dot(v, v)
    t = 0.0 if L2 == 0.0 else np.clip(np.dot(c - p0, v) / L2, 0.0, 1.0)
    closest = p0 + t * v

    return np.linalg.norm(c - closest) - obs[2]

for delta in [0, 2, 5, 10, 20]:
    scene.reseed(SCENE_SEED=SCENE_SEED)
    for i in range(500):

        sens.reseed(SENSE_SEED+i)
        
        ##### INIT #####
        po_x, po_y, r_o = scene.obstacle()
        x0 = scene.initial_condition()

        xg = scene.terminal_condition()
        gap = corridor_gap(x0,xg,np.array([po_x, po_y, r_o]))
        R_min = x0[3]**2 / (g * np.tan(limits.PHI_MAX))

        a_0, t_phi_0 = scene.initial_control(x=x0,
                                            goal=xg)

        obs = obstacle.Obstacle(po_x=po_x, po_y=po_y, po_r=r_o, delta=delta)

        sol.set_obstacles(obstacles=[obs])
        sol.reset(tphi0=t_phi_0)

        # Record skeleton. Every branch writes the same keys.
        output[i] = {
            'scenario_condition': None,
            'initial_condition': x0.tolist(),
            'terminal condition': xg.tolist(),
            'obstacle_condition': [po_x, po_y, r_o],
            'gap_norm': gap/r_o,
            'offset_norm': scene.offset/r_o,
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

        if np.linalg.norm(x0[:2] - np.array([po_x, po_y])) <= r_o:
            output[i]['scenario_condition'] = 'fail; init inside obstacle'
            continue
        
        if np.linalg.norm(xg[:2] - np.array([po_x, po_y])) <= r_o:
            output[i]['scenario_condition'] = 'fail; term inside obstacle'
            continue

        if np.linalg.norm(np.array([po_x, po_y]) - x0[:2]) < r_o + R_min:
            output[i]['scenario_condition'] = 'fail; obstacle inside minimum turn radius'
            continue


        t, traj, uncert_inputs, inputs, slacks, _, _ = rk45.simulate_with_filter(f=vehicle.f, 
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
            h.append(obs.h(traj[j,:]))

        print(f"min h = {min(h):.6f}")
        print(f"min h at t = {t[np.argmin(h)]:.2f} s")
        print(f"clearance = {(np.sqrt(min(h) + r_o**2) - r_o):.2f} m")
        print(f"path length = {helpers.path_length(px, py):.2f} m")
        print(f"max |a| = {np.max(np.abs(a)):.2f} m/s^2")
        print(f"max |phi| = {np.max(np.abs(phi)):.2f} rads")
        print(f"max slack = {np.max(slacks):.2f}")

        d_min = np.min(np.linalg.norm(traj[:, :2] - np.array([po_x, po_y]), axis=1))
        output[i]['min |h|'] = min(h)
        output[i]['clearance'] = d_min - obs.po_r_true
        output[i]['path length'] = helpers.path_length(px, py)
        output[i]['max |a|'] = np.max(np.abs(a))
        output[i]['max |phi|'] = np.max(np.abs(phi))
        output[i]['max_da'] = float(np.max(np.abs(a - a_nom)))
        output[i]['max_dphi'] = float(np.max(np.abs(phi - phi_nom)))
        output[i]['max slack'] = np.max(slacks)
        output[i]['violation'] = bool(d_min < obs.po_r_true)
        output[i]['guarantee_void'] = bool(np.max(slacks) > 1e-6)
        output[i]['margin_breach'] = bool(d_min < r_o + obs.delta)

        if np.linalg.norm(traj[-1, :2]-xg[:2]) > 5.0:
            output[i]['scenario_condition'] = f'terminal condition not reached in {tf} s'
        else:
            output[i]['scenario_condition'] = 'pass'
            

    with open(f'output_inflated_rad_{delta}m.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)