import numpy as np
import pytest
from src import limits
from src.vehicle import f

G = 9.81

def test_trim_is_not_an_equilibrium(trim_state, zero_input):
    """Position moves, heading and speed hold."""
    xdot = f(0, trim_state, zero_input)
    assert xdot[0] == pytest.approx(limits.V_CRUISE)
    assert xdot[1] == pytest.approx(0.0, abs=1e-12)
    assert xdot[2] == pytest.approx(0.0, abs=1e-12)
    assert xdot[3] == pytest.approx(0.0, abs=1e-12)

@pytest.mark.parametrize("psi", [0.0, np.pi/4, np.pi/2, np.pi, -np.pi/3])
def test_velocity_is_along_heading(psi, zero_input):
    """Catches a swapped sin/cos or a transposed frame."""
    x = np.array([0.0, 0.0, psi, limits.V_CRUISE])
    xdot = f(0, x, zero_input)
    speed = np.hypot(xdot[0], xdot[1])
    assert speed == pytest.approx(limits.V_CRUISE)
    assert np.arctan2(xdot[1], xdot[0]) == pytest.approx(psi)

def test_positive_bank_turns_left(trim_state):
    """Sign convention. Right-handed frame: +phi -> +psi_dot."""
    u = np.array([0.0, np.tan(0.3)])
    assert f(0, trim_state, u)[2] > 0.0

def test_yaw_rate_matches_coordinated_turn(trim_state):
    phi = 0.4
    u = np.array([0.0, np.tan(phi)])
    expected = G * np.tan(phi) / limits.V_CRUISE
    assert f(0, trim_state, u)[2] == pytest.approx(expected)

def test_yaw_rate_scales_inversely_with_speed():
    """Slower flight turns tighter. This coupling drives the whole build."""
    u = np.array([0.0, np.tan(limits.PHI_MAX)])
    slow = f(0, np.array([0., 0., 0., limits.V_STALL]), u)[2]
    fast = f(0, np.array([0., 0., 0., limits.V_MAX]), u)[2]
    assert slow > fast

def test_acceleration_enters_speed_only(trim_state):
    u = np.array([1.5, 0.0])
    assert f(0, trim_state, u)[3] == pytest.approx(1.5)

def test_dynamics_does_not_mutate_state(trim_state, zero_input):
    """f MUST be pure. An in-place write here corrupts every RK4 stage."""
    before = trim_state.copy()
    f(0, trim_state, zero_input)
    np.testing.assert_array_equal(trim_state, before)