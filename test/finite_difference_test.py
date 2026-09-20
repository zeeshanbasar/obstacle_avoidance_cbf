import numpy as np
from src import obstacle, limits, vehicle

SEED = 42
rng = np.random.default_rng(SEED)

dt = 1e-6

o = obstacle.Obstacle(po_x=0.0,
                      po_y=0.0,
                      po_r=50.0)

for i in range(200):
    x = rng.uniform(low=np.array([-2000.0, -2000.0, -np.pi, limits.V_STALL]),
                high=np.array([2000.0, 2000.0, np.pi, limits.V_MAX]))

    u = rng.uniform(low=np.array([-limits.A_MAX, -np.tan(limits.PHI_MAX)]),
                    high=np.array([limits.A_MAX, np.tan(limits.PHI_MAX)]))

    L = (o.h(x + dt*vehicle.f(0.0, x, u)) - o.h(x)) / dt

    R = o.h_dot(x)

    if np.abs(L-R) > 1e-4:
        print(f"finite difference failed at sample {i}:")
        print(f"x = {x}")
        print(f"u = {u}")

        

    