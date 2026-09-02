import numpy as np
import pytest

from src import limits, helpers
from src.vehicle import f
from src.rk45 import simulate
from src.guidance import nominal

G = 9.81

R_MIN_CRUISE = limits.V_CRUISE**2 / (G * np.tan(limits.PHI_MAX))
R_BLEND = 4.0 * R_MIN_CRUISE


# ---------- helpers ----------

def state(x=0.0, y=0.0, psi=0.0, V=None):
    return np.array([x, y, psi, limits.V_CRUISE if V is None else V])


def goal_at(x, y, psi_f=0.0, V_g=None):
    return np.array([x, y, psi_f, limits.V_CRUISE if V_g is None else V_g])


def inputs_along(traj, goal):
    """Recompute the commands. Valid because nominal is pure and held zero-order."""
    return np.array([nominal(x, goal) for x in traj])


@pytest.fixture
def far_goal():
    """Far enough ahead that the blend is inactive (w = 1)."""
    return goal_at(600.0, 0.0)


# ---------- heading_wrap ----------

@pytest.mark.parametrize("a,expected", [
    (0.0, 0.0), (np.pi/2, np.pi/2), (-np.pi/2, -np.pi/2),
    (3*np.pi, -np.pi), (2*np.pi + 0.3, 0.3), (-2*np.pi - 0.3, -0.3),
])
def test_heading_wrap(a, expected):
    assert helpers.heading_wrap(a) == pytest.approx(expected, abs=1e-12)


def test_heading_wrap_range():
    a = np.linspace(-20.0, 20.0, 4001)
    w = helpers.heading_wrap(a)
    assert np.all(w > -np.pi - 1e-12) and np.all(w <= np.pi + 1e-12)


# ---------- open loop: one call, no integration ----------

def test_goal_ahead_gives_zero_bank(far_goal):
    a, tphi = nominal(state(), far_goal)
    assert tphi == pytest.approx(0.0, abs=1e-9)


def test_goal_left_banks_left():
    _, tphi = nominal(state(psi=0.0), goal_at(0.0, 600.0, np.pi/2))
    assert tphi > 0.0


def test_goal_right_banks_right():
    _, tphi = nominal(state(psi=0.0), goal_at(0.0, -600.0, -np.pi/2))
    assert tphi < 0.0


def test_large_error_saturates_bank():
    _, tphi = nominal(state(psi=0.0), goal_at(-600.0, 1.0, np.pi))
    assert abs(np.arctan(tphi)) == pytest.approx(limits.PHI_MAX, rel=1e-9)


def test_small_error_does_not_saturate(far_goal):
    _, tphi = nominal(state(psi=np.deg2rad(-10.0)), far_goal)
    assert 0.0 < np.arctan(tphi) < limits.PHI_MAX


def test_blend_crosses_wrap_boundary():
    """The blend MUST act on the wrapped difference, not on raw angles.

    Bearing +3.0 and terminal -3.0 are 0.28 rad apart. Blended halfway the
    reference is near +/-pi. Averaging the raw angles gives 0.0, off by pi.
    """
    psi_bearing, psi_f = 3.0, -3.0
    rng = 0.5 * R_BLEND
    px = -rng * np.cos(psi_bearing)
    py = -rng * np.sin(psi_bearing)

    x = state(x=px, y=py, psi=-np.pi + 0.3)
    _, tphi = nominal(x, goal_at(0.0, 0.0, psi_f))

    phi = np.arctan(tphi)
    assert abs(phi) < limits.PHI_MAX - 1e-6, "reference is off by pi: raw-angle blend"
    assert phi == pytest.approx(-0.3, abs=0.05)


def test_blend_weight_endpoints():
    """Far away follow the bearing; at the goal follow the terminal heading."""
    _, tphi_far = nominal(state(psi=0.0), goal_at(10.0 * R_BLEND, 0.0, np.pi/2))
    assert abs(np.arctan(tphi_far)) < np.deg2rad(1.0)

    _, tphi_near = nominal(state(psi=0.0), goal_at(0.5, 0.0, np.pi/2))
    assert np.arctan(tphi_near) > np.deg2rad(20.0)


def test_speed_command_signs(far_goal):
    a_slow, _ = nominal(state(V=limits.V_STALL), far_goal)
    a_fast, _ = nominal(state(V=limits.V_MAX), far_goal)
    a_trim, _ = nominal(state(V=limits.V_CRUISE), far_goal)
    assert a_slow > 0.0 and a_fast < 0.0
    assert a_trim == pytest.approx(0.0, abs=1e-9)
    assert abs(a_slow) <= limits.A_MAX + 1e-12
    assert abs(a_fast) <= limits.A_MAX + 1e-12


def test_guidance_does_not_mutate(far_goal):
    x, g = state(psi=0.7, V=15.0), far_goal.copy()
    x_before, g_before = x.copy(), g.copy()
    nominal(x, g)
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(g, g_before)


# ---------- closed loop ----------

@pytest.mark.parametrize("psi0", [0.0, np.pi/2, -np.pi/2, np.pi - 0.05])
def test_reaches_goal_from_any_heading(psi0):
    g = goal_at(400.0, 0.0)
    traj = simulate(f, state(x=-100.0, psi=psi0), g, nominal, 0.02, T=60.0)
    miss = np.min(np.linalg.norm(traj[:, :2] - g[:2], axis=1))
    assert miss < 5.0


@pytest.mark.parametrize("psi0", [0.0, np.pi/2, -np.pi/2, np.pi - 0.05])
def test_bank_never_exceeds_limit(psi0):
    g = goal_at(400.0, 0.0)
    traj = simulate(f, state(x=-100.0, psi=psi0), g, nominal, 0.02, T=60.0)
    phi = np.arctan(inputs_along(traj, g)[:, 1])
    assert np.max(np.abs(phi)) <= limits.PHI_MAX + 1e-9


def test_acceleration_never_exceeds_limit():
    g = goal_at(400.0, 0.0)
    traj = simulate(f, state(x=-100.0, V=limits.V_STALL), g, nominal, 0.02, T=60.0)
    assert np.max(np.abs(inputs_along(traj, g)[:, 0])) <= limits.A_MAX + 1e-9


def test_speed_returns_to_cruise():
    g = goal_at(800.0, 0.0)
    traj = simulate(f, state(V=limits.V_STALL), g, nominal, 0.02, T=20.0)
    assert traj[-1, 3] == pytest.approx(limits.V_CRUISE, abs=0.5)


def test_speed_stays_inside_envelope():
    g = goal_at(800.0, 0.0)
    traj = simulate(f, state(V=limits.V_STALL), g, nominal, 0.02, T=30.0)
    assert np.min(traj[:, 3]) >= limits.V_STALL - 1e-6
    assert np.max(traj[:, 3]) <= limits.V_MAX + 1e-6


def test_terminal_heading_is_approached():
    psi_f = np.pi / 2
    g = goal_at(400.0, 0.0, psi_f)
    traj = simulate(f, state(x=-100.0), g, nominal, 0.02, T=90.0)
    k = int(np.argmin(np.linalg.norm(traj[:, :2] - g[:2], axis=1)))
    assert abs(helpers.heading_wrap(traj[k, 2] - psi_f)) < np.deg2rad(45.0)