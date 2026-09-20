# obstacle-avoidance-cbf

A Control Barrier Function (CBF) safety filter for obstacle avoidance on a 3-DOF fixed-wing UAV.

The filter sits between a nominal guidance law and the vehicle. It takes the guidance command, solves a small quadratic program (QP), and returns the closest command that keeps the vehicle outside a circular obstacle.

## Model

States are `[px, py, psi, V]`. Inputs are `[a, tan(phi)]`.

| Symbol | Meaning | Unit |
|---|---|---|
| `px`, `py` | position | m |
| `psi` | heading | rad |
| `V` | airspeed | m/s |
| `a` | longitudinal acceleration | m/s² |
| `tan(phi)` | tangent of bank angle | - |

The vehicle uses coordinated-turn kinematics: `psi_dot = g*tan(phi)/V`. The input is `tan(phi)` and not `phi`. This substitution keeps the QP linear.

## Install

The project needs Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

## Run the tests

```bash
pytest
```

## Run a scenario

Experiment runners are in `scripts/`. Each script writes its results to a JSON file.

```bash
python scripts/<runner>.py
```

## Layout

```
src/
  limits.py       tuning constants and noise levels
  helpers.py      heading wrap, path length
  vehicle.py      3-DOF coordinated-turn dynamics
  rk45.py         RK4 integrator and the two simulation loops
  guidance.py     nominal guidance law
  obstacle.py     barrier function and the CBF constraint row
  filter.py       the CBF-QP safety filter
  estimator.py    sensor model: noise, zero-order hold, dropouts
  scenario_gen.py randomized scenario generator

test/             pytest suite
  conftest.py
  test_estimator.py
  test_vehicle.py
  test_integrate.py
  test_guidance.py
  test_filter.py
  test_obstacle.py
  test_filter_multi.py

scripts/          experiment runners
  basic_avoidance_test.py
  basic_guidance_test.py
  basic_sensor_test.py
  finite_difference_test.py
  constant_turn.py
  zero_bank_zero_acceleration.py
  monte_carlo_test.py
  sweep_test.py
  mc_with_sensor_test.py
  mc_with_sensor_test_inflation.py
  mc_test_rate_sweep.py
  mc_test_fix_rate_delta_sweep.py
  mc_test_dropout.py
  corridor_test.py

```

## Minimal use

```python
import numpy as np
from src.vehicle import f
from src.guidance import nominal
from src.obstacle import Obstacle
from src.filter import Solver
from src.rk45 import simulate_with_filter
from src.estimator import Sensor

obs = [Obstacle(po_x=200.0, po_y=0.0, po_r=50.0, delta=2.0)]
sol = Solver(x_dim=4, u_dim=2, dt=0.02)
sol.set_obstacles(obs)          # required before the first solve
sens = Sensor(isNoise=True, rate_hz=50.0)

x0 = np.array([0.0, 0.0, 0.0, 18.0])
xg = np.array([600.0, 0.0, 0.0, 18.0])

out = simulate_with_filter(f, sol, x0, xg, nominal, h=0.02, T=60.0, sens=sens)
```

`Solver.set_obstacles` must be called before the first `Solver.solve`. `set_obstacles` method builds the constraint padding that `solve` need to build the QP once and reuse it during the Monte Carlo experiments.

## Declaration of AI use

Claude Sonnet/Opus by Anthropic was used for project management, and to write the pytest files. After using this tool, I reviewed and edited the content as needed and take full responsibility for the content of the publication. All em-dashes are human generated.
 
## License
 
See `LICENSE`.
