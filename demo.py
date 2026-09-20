import os
import numpy as np
import matplotlib.pyplot as plt

from src import limits, guidance
from src.vehicle import f
from src.obstacle import Obstacle
from src.filter import Solver
from src.rk45 import simulate_with_filter
from src.estimator import Sensor

DT = 0.02
T_MAX = 60.0
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

def make_circle(cx, cy, r, n=200):
    th = np.linspace(0, 2 * np.pi, n)
    return cx + r * np.cos(th), cy + r * np.sin(th)


def single_obs():

    obs = [Obstacle(po_x=250.0, po_y=0.0, po_r=50.0, delta=0.0)]
    x0 = np.array([0.0,0.0,0.0,limits.V_CRUISE])
    xg = np.array([500.0, 60.0, 0.0, limits.V_CRUISE])

    sens = Sensor(isNoise=False, isDrop=False, rate_hz=50.0)

    sol = Solver(x_dim=4, u_dim=2, obs=obs, dt=DT, max_n_obs=len(obs))

    time, traj, uncert_u, cert_u, slacks, x_hat, age = simulate_with_filter(f=f, 
                                                                            sol=sol,
                                                                            x0=x0,
                                                                            xg=xg,
                                                                            u0=guidance.nominal, 
                                                                            h=DT, 
                                                                            T=T_MAX, 
                                                                            sens=sens)

    h_vals = np.stack([np.array([o.h(x) for x in traj]) for o in obs], axis=1)

    output = {
        "time": time,
        "traj": traj,
        "uncert_u": uncert_u,
        "cert_u": cert_u,
        "slacks": slacks,
        "h_vals": h_vals,
        "obstacles": obs,
        "xg": xg,
        "slack_tolerance": sol.slack_tolerance,
        "n_slack_failures": len([slacks >= sol.slack_tolerance]),
        "n_steps": len(time),
    }

    title = f"Single obstacle — CBF-QP filtered avoidance\nNominal case — no sensor noise, delay or dropout, non-inflated obstacle."
    # title = f"Single-obstacle corridor — CBF-QP filtered avoidance\nNoisy and delayed sensor data with random dropouts."
    # fname = f"single_inflated_obstacle_{10.0}m_sensorSampleRate_{5}Hz_isNoisy_isDropping.png"
    fname = f"single_obstacle_nominal.png"

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)
 
    # --- Trajectory plot ---
    ax = axs[0, 0]
    ax.plot(traj[:, 0], traj[:, 1], "b-", lw=2, label="Certified trajectory")
    for i, obs in enumerate(obs):
        cx, cy = make_circle(obs.po_x, obs.po_y, obs.po_r_true)
        ax.plot(cx, cy, "r-", lw=1.5, label="Obstacle (true)" if i == 0 else None)
        if obs.delta != 0.0:
            cxi, cyi = make_circle(obs.po_x, obs.po_y, obs.po_r)
            ax.plot(cxi, cyi, "r--", lw=1, alpha=0.6,
                     label="Obstacle (inflated)" if i == 0 else None)
    ax.plot(traj[0, 0], traj[0, 1], "go", ms=8, label="Start")
    ax.plot(xg[0], xg[1], "k*", ms=14, label="Goal")
    ax.set_xlabel("px [m]")
    ax.set_ylabel("py [m]")
    ax.set_title("Trajectory")
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
 
    # --- h(x) plot ---
    ax = axs[0, 1]
    for i in range(h_vals.shape[1]):
        ax.plot(time, h_vals[:, i], label=f"h obstacle {i}")
    ax.axhline(0.0, color="k", lw=1, ls="--", label="h = 0 (boundary)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("h(x)")
    ax.set_title("Barrier value (h >= 0 required)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
 
    # --- Slack plot ---
    slacks = np.atleast_2d(output["slacks"])
    if slacks.shape[0] == len(time):
        pass
    else:
        slacks = slacks.reshape(len(time), -1)

    ax = axs[1, 0]
    n_obs_real = h_vals.shape[1]
    for i in range(min(n_obs_real, slacks.shape[1])):
        ax.plot(time, slacks[:, i], label=f"slack obstacle {i}")
    ax.axhline(sol.slack_tolerance, color="k", lw=1, ls="--", label="tolerance")
    max_slack = np.max(slacks[:, :n_obs_real]) if slacks.size else 0.0
    guarantee_holds = max_slack <= sol.slack_tolerance
    ax.set_xlabel("t [s]")
    ax.set_ylabel("slack")
    verdict = "GUARANTEE HOLDS (slack ~ 0)" if guarantee_holds else "GUARANTEE VOID (slack > tol)"
    ax.set_title(f"CBF slack — {verdict}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
 
    # --- Control plot ---
    ax = axs[1, 1]
    ax.plot(time, uncert_u[:, 0], "b--", alpha=0.5, label="nominal a")
    ax.plot(time, cert_u[:, 0], "b-", label="certified a")
    ax.plot(time, uncert_u[:, 1], "r--", alpha=0.5, label="nominal tan(phi)")
    ax.plot(time, cert_u[:, 1], "r-", label="certified tan(phi)")
    ax.axhline(limits.A_MAX, color="gray", lw=0.7, ls=":")
    ax.axhline(-limits.A_MAX, color="gray", lw=0.7, ls=":")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("control value")
    ax.set_title("Nominal vs. certified control")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
 
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    n_fail = output["n_slack_failures"]
    n_steps = output["n_steps"]
    print(f"[saved] {path}  (max slack = {max_slack:.4g}, tol = {sol.slack_tolerance:.1e}, "
          f"{n_fail}/{n_steps} steps over tolerance) -> {verdict}")

def multi_obs():

    obs1 = Obstacle(po_x=150.0, po_y=30.0, po_r=45.0, delta=0.0)
    obs2 = Obstacle(po_x=350.0, po_y=-30.0, po_r=45.0, delta=0.0)
    obs = [obs1, obs2]
 
    x0 = np.array([0.0, 0.0, 0.0, limits.V_CRUISE])
    xg = np.array([500.0, 0.0, 0.0, limits.V_CRUISE])


    sens = Sensor(isNoise=False, isDrop=False, rate_hz=50.0)

    sol = Solver(x_dim=4, u_dim=2, obs=obs, dt=DT, max_n_obs=len(obs))

    time, traj, uncert_u, cert_u, slacks, x_hat, age = simulate_with_filter(f=f, 
                                                                            sol=sol,
                                                                            x0=x0,
                                                                            xg=xg,
                                                                            u0=guidance.nominal, 
                                                                            h=DT, 
                                                                            T=T_MAX, 
                                                                            sens=sens)

    h_vals = np.stack([np.array([o.h(x) for x in traj]) for o in obs], axis=1)

    output = {
        "time": time,
        "traj": traj,
        "uncert_u": uncert_u,
        "cert_u": cert_u,
        "slacks": slacks,
        "h_vals": h_vals,
        "obstacles": obs,
        "xg": xg,
        "slack_tolerance": sol.slack_tolerance,
        "n_slack_failures": len([slacks >= sol.slack_tolerance]),
        "n_steps": len(time),
    }

    title = f"Multi-obstacle corridor — CBF-QP filtered avoidance\nNominal case — no sensor noise, delay or dropout, non-inflated obstacle."
    # title = f"Multi-obstacle corridor — CBF-QP filtered avoidance\nNoisy and delayed sensor data with random dropouts."
    # fname = f"multi_inflated_obstacle_{10.0}m_sensorSampleRate_{5}Hz_isNoisy_isDropping.png"
    fname = f"multi_obstacle_nominal.png"

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)
    
    # --- Trajectory plot ---
    ax = axs[0, 0]
    ax.plot(traj[:, 0], traj[:, 1], "b-", lw=2, label="Certified trajectory")
    for i, obs in enumerate(obs):
        cx, cy = make_circle(obs.po_x, obs.po_y, obs.po_r_true)
        ax.plot(cx, cy, "r-", lw=1.5, label="Obstacle (true)" if i == 0 else None)
        if obs.delta != 0.0:
            cxi, cyi = make_circle(obs.po_x, obs.po_y, obs.po_r)
            ax.plot(cxi, cyi, "r--", lw=1, alpha=0.6,
                        label="Obstacle (inflated)" if i == 0 else None)
    ax.plot(traj[0, 0], traj[0, 1], "go", ms=8, label="Start")
    ax.plot(xg[0], xg[1], "k*", ms=14, label="Goal")
    ax.set_xlabel("px [m]")
    ax.set_ylabel("py [m]")
    ax.set_title("Trajectory")
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    
    # --- h(x) plot ---
    ax = axs[0, 1]
    for i in range(h_vals.shape[1]):
        ax.plot(time, h_vals[:, i], label=f"h obstacle {i}")
    ax.axhline(0.0, color="k", lw=1, ls="--", label="h = 0 (boundary)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("h(x)")
    ax.set_title("Barrier value (h >= 0 required)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    
    # --- Slack plot ---
    slacks = np.atleast_2d(output["slacks"])
    if slacks.shape[0] == len(time):
        pass
    else:
        slacks = slacks.reshape(len(time), -1)

    ax = axs[1, 0]
    n_obs_real = h_vals.shape[1]
    for i in range(min(n_obs_real, slacks.shape[1])):
        ax.plot(time, slacks[:, i], label=f"slack obstacle {i}")
    ax.axhline(sol.slack_tolerance, color="k", lw=1, ls="--", label="tolerance")
    max_slack = np.max(slacks[:, :n_obs_real]) if slacks.size else 0.0
    guarantee_holds = max_slack <= sol.slack_tolerance
    ax.set_xlabel("t [s]")
    ax.set_ylabel("slack")
    verdict = "GUARANTEE HOLDS (slack ~ 0)" if guarantee_holds else "GUARANTEE VOID (slack > tol)"
    ax.set_title(f"CBF slack — {verdict}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    
    # --- Control plot ---
    ax = axs[1, 1]
    ax.plot(time, uncert_u[:, 0], "b--", alpha=0.5, label="nominal a")
    ax.plot(time, cert_u[:, 0], "b-", label="certified a")
    ax.plot(time, uncert_u[:, 1], "r--", alpha=0.5, label="nominal tan(phi)")
    ax.plot(time, cert_u[:, 1], "r-", label="certified tan(phi)")
    ax.axhline(limits.A_MAX, color="gray", lw=0.7, ls=":")
    ax.axhline(-limits.A_MAX, color="gray", lw=0.7, ls=":")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("control value")
    ax.set_title("Nominal vs. certified control")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    n_fail = output["n_slack_failures"]
    n_steps = output["n_steps"]
    print(f"[saved] {path}  (max slack = {max_slack:.4g}, tol = {sol.slack_tolerance:.1e}, "
            f"{n_fail}/{n_steps} steps over tolerance) -> {verdict}")

def main():
    # print("Running single-obstacle scenario...")
    # single_obs()
    print("Running multi-obstacle scenario...")
    multi_obs()
    # print(f"\nDone. Figures written to: {OUT_DIR}")
 
 
if __name__ == "__main__":
    main()
