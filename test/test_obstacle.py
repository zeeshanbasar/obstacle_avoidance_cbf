import numpy as np
import pytest

from src import limits, vehicle
from src.obstacle import Obstacle

STEP = 1e-3          # central difference is exact here; larger step = less roundoff
RTOL = 1e-8


@pytest.fixture
def obs():
    return Obstacle(po_x=0.0, po_y=0.0, po_r=50.0)


def random_states(n, seed=42):
    """Positions bounded to +/-300 m. Beyond that h is order 1e6 and the
    finite difference loses digits to cancellation for no added coverage."""
    rng = np.random.default_rng(seed)
    x = rng.uniform([-300.0, -300.0, -np.pi, limits.V_STALL],
                    [300.0, 300.0, np.pi, limits.V_MAX], size=(n, 4))
    u = rng.uniform([-limits.A_MAX, -np.tan(limits.PHI_MAX)],
                    [limits.A_MAX, np.tan(limits.PHI_MAX)], size=(n, 2))
    return x, u


def test_h_sign(obs):
    assert obs.h(np.array([0.0, 0.0, 0.0, 18.0])) < 0.0        # inside
    assert obs.h(np.array([50.0, 0.0, 0.0, 18.0])) == pytest.approx(0.0)
    assert obs.h(np.array([200.0, 0.0, 0.0, 18.0])) > 0.0       # outside


def test_h_dot_matches_finite_difference(obs):
    """h_dot MUST equal grad(h) . f(x, u). Catches every frame and sign error."""
    X, U = random_states(200)
    for x, u in zip(X, U):
        fx = vehicle.f(0.0, x, u)
        fd = (obs.h(x + STEP * fx) - obs.h(x - STEP * fx)) / (2.0 * STEP)
        assert fd == pytest.approx(obs.h_dot(x), rel=RTOL, abs=1e-9)


def test_h_dot_is_independent_of_input(obs):
    """Relative degree 2: no input appears in h_dot. This is the fact the
    whole HOCBF construction rests on, so it MUST be tested, not assumed."""
    X, _ = random_states(50, seed=7)
    for x in X:
        u_a = np.array([limits.A_MAX, np.tan(limits.PHI_MAX)])
        u_b = np.array([-limits.A_MAX, -np.tan(limits.PHI_MAX)])
        fa = (obs.h(x + STEP * vehicle.f(0.0, x, u_a))
              - obs.h(x - STEP * vehicle.f(0.0, x, u_a))) / (2.0 * STEP)
        fb = (obs.h(x + STEP * vehicle.f(0.0, x, u_b))
              - obs.h(x - STEP * vehicle.f(0.0, x, u_b))) / (2.0 * STEP)
        assert fa == pytest.approx(fb, rel=RTOL, abs=1e-9)


def test_h_dot_sign_is_physical(obs):
    """Flying away from the centre MUST increase h."""
    outbound = np.array([100.0, 0.0, 0.0, 18.0])          # heading +x, obstacle behind
    inbound = np.array([100.0, 0.0, np.pi, 18.0])
    tangent = np.array([100.0, 0.0, np.pi / 2, 18.0])
    assert obs.h_dot(outbound) > 0.0
    assert obs.h_dot(inbound) < 0.0
    assert obs.h_dot(tangent) == pytest.approx(0.0, abs=1e-9)

def test_no_deadlock(obs):
    """(e·d)**2 + (e·d_perp)**2 = norm(e)**2, both coeffs cannot become 0 together."""
    X, U = random_states(200)
    for x, u in zip(X, U):
        A, _ = obs.constraint(x, u)
        assert np.linalg.norm(A) > 0.0

def test_finite_difference_constraints(obs):
    _STEP = 1e-5

    X, U = random_states(200)
    for x, u in zip(X, U):
        fx = vehicle.f(0.0, x, u)
        fd = (obs.h_dot(x + _STEP*fx) - obs.h_dot(x - _STEP*fx))/(2.0*_STEP)

        A, _ = obs.constraint(x, u)

        assert A @ u + 2*(x[-1]**2) == pytest.approx(fd, rel=RTOL, abs=1e-9)

def test_finite_difference_constraints_2(obs):
    _STEP = 1e-5

    X, U = random_states(200)
    for x, u in zip(X, U):
        fx = vehicle.f(0.0, x, u)
        fd = (obs.h(x + _STEP*fx) - obs.h(x - _STEP*fx))/(2.0*_STEP)

        A, b = obs.constraint(x, u)

        psi1 = obs.psi_1(x)
        psi1_dot = (obs.psi_1(x + _STEP*fx) - obs.psi_1(x - _STEP*fx))/(2.0*_STEP)

        assert (A @ u) - b == pytest.approx(psi1_dot + limits.K_1*psi1, rel=RTOL, abs=1e-9)

def test_inflated_radius_shifts_the_zero():
    obs = Obstacle(po_x=0.0, po_y=0.0, po_r=50.0, delta=5.0)
    assert obs.h(np.array([55.0, 0.0, 0.0, 18.0])) == pytest.approx(0.0)
    assert obs.h(np.array([52.0, 0.0, 0.0, 18.0])) < 0.0    # inside the margin annulus
    assert obs.h(np.array([60.0, 0.0, 0.0, 18.0])) > 0.0