import numpy as np
import pytest
from src import limits
from src.vehicle import f
from src.rk45 import simulate
from conftest import const, DUMMY_GOAL

G = 9.81

def test_straight_flight(trim_state, zero_input, dt):
    T = 10.0
    traj = simulate(f, trim_state, DUMMY_GOAL, const(zero_input), dt,
                    nSteps=round(T / dt))
    end = traj[-1]
    assert end[0] == pytest.approx(limits.V_CRUISE * T, rel=1e-9)
    assert end[1] == pytest.approx(0.0, abs=1e-9)
    assert end[3] == pytest.approx(limits.V_CRUISE, rel=1e-12)

@pytest.mark.parametrize("phi", [0.2, 0.4, limits.PHI_MAX])
@pytest.mark.parametrize("V", [limits.V_STALL, limits.V_CRUISE, limits.V_MAX])
def test_turn_radius(phi, V, dt):
    """R = V^2 / (g tan phi). The test that catches frame and sign errors."""
    R = V**2 / (G * np.tan(phi))
    period = 2 * np.pi * R / V
    x0 = np.array([0.0, 0.0, 0.0, V])
    u = np.array([0.0, np.tan(phi)])
    traj = simulate(f, x0, DUMMY_GOAL, const(u), dt, nSteps=round(period/dt))

    # Centre lies one radius to the left of the start, for positive bank.
    centre = np.array([0.0, R])
    radii = np.linalg.norm(traj[:, :2] - centre, axis=1)
    assert np.max(np.abs(radii - R)) / R < 1e-2

def test_turn_closes_the_circle(dt):
    V, phi = limits.V_CRUISE, 0.4
    R = V**2 / (G * np.tan(phi))
    T = 2 * np.pi * R / V
    x0 = np.array([0.0, 0.0, 0.0, V])
    u = np.array([0.0, np.tan(phi)])
    traj = simulate(f, x0, DUMMY_GOAL, const(u), dt, nSteps=round(T / dt))
    assert np.linalg.norm(traj[-1, :2] - traj[0, :2]) / R < 1e-2

def test_rk4_order_of_accuracy():
    V, phi = limits.V_CRUISE, 0.4
    omega = G * np.tan(phi) / V
    R = V / omega
    t_end = (np.pi / 2) / omega
    x0 = np.array([0.0, 0.0, 0.0, V])
    u = np.array([0.0, np.tan(phi)])
    exact = np.array([R * np.sin(omega * t_end), R * (1.0 - np.cos(omega * t_end))])

    errors = []
    for n in (64, 128):
        traj = simulate(f, x0, DUMMY_GOAL, const(u), t_end / n, nSteps=n)
        errors.append(np.linalg.norm(traj[-1, :2] - exact))

    assert errors[0] / errors[1] > 8.0