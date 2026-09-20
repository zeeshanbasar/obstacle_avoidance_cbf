import numpy as np
import pytest

from src import limits
from src.obstacle import Obstacle
from src.filter import Solver

DT = 0.02
TAN_MAX = np.tan(limits.PHI_MAX)


@pytest.fixture
def sol():
    return Solver(x_dim=4, u_dim=2,
                  obs=[Obstacle(po_x=0.0, po_y=0.0, po_r=50.0)], dt=DT)


def random_states(n, seed=11, lo=-400.0, hi=400.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform([lo, lo, -np.pi, limits.V_STALL],
                    [hi, hi, np.pi, limits.V_MAX], size=(n, 4))
    u = rng.uniform([-limits.A_MAX, -TAN_MAX],
                    [limits.A_MAX, TAN_MAX], size=(n, 2))
    return x, u


# ---------- construction ----------

def test_problem_is_built_once(sol):
    """The QP MUST NOT grow across calls. This is the original failure: a
    global Opti that accumulated variables and constraints every timestep."""
    opti = sol.opti_dict['opti']
    before = (opti.nx, opti.ng, opti.np)
    x = np.array([-800.0, 300.0, 0.0, 18.0])
    for _ in range(50):
        sol.solve(x, np.array([0.5, 0.2]))
    assert (opti.nx, opti.ng, opti.np) == before


# ---------- the filter is a filter ----------

def test_no_op_far_from_obstacle(sol):
    """Far away the command MUST pass through untouched.

    reset(tphi0=u_nom[1]) centres the rate box on the nominal bank. Without
    it the rate box alone clips bank to tan(PHI_DOT_MAX*dt) = 0.021 on the
    first call, and this measures the actuator model, not the barrier.
    """
    u_nom = np.array([0.5, 0.2])
    sol.reset(tphi0=u_nom[1])
    u, s = sol.solve(np.array([-800.0, 300.0, 0.0, 18.0]), u_nom)
    np.testing.assert_allclose(u, u_nom, atol=1e-8)
    assert s == pytest.approx(0.0, abs=1e-9)


def test_barrier_row_is_slack_far_away(sol):
    """Independent of the QP. The row itself MUST admit the nominal command
    at long range, or no gain setting will make the filter transparent."""
    X, U = random_states(100, lo=-2000.0, hi=2000.0)
    for x, u_nom in zip(X, U):
        if np.hypot(x[0], x[1]) < 600.0:
            continue
        A, b = sol.obs[0].constraint(x, u_nom)
        assert A @ u_nom - b > 0.0


# ---------- limits ----------

def test_output_inside_input_box(sol):
    X, U = random_states(150)
    for x, u_nom in zip(X, U):
        sol.reset()
        u, _ = sol.solve(x, u_nom)
        assert abs(u[0]) <= limits.A_MAX + 1e-8
        assert abs(u[1]) <= TAN_MAX + 1e-8


def test_roll_rate_box_is_respected(sol):
    """Consecutive bank commands MUST NOT differ by more than PHI_DOT_MAX*dt.
    Fails if lo/hi are baked in at construction instead of set per solve."""
    sol.reset(tphi0=0.0)
    x = np.array([-200.0, 40.0, 0.0, 18.0])
    u_nom = np.array([0.0, TAN_MAX])          # demand full bank immediately
    phi_prev = 0.0
    for _ in range(30):
        u, _ = sol.solve(x, u_nom)
        phi = np.arctan(u[1])
        assert abs(phi - phi_prev) <= limits.PHI_DOT_MAX * DT + 1e-8
        phi_prev = phi


# ---------- robustness ----------

def test_solve_never_raises_inside_obstacle(sol):
    """h < 0 makes the row unsatisfiable. Slack MUST absorb it and the filter
    MUST still return a command. An infeasible QP in flight is a dropout."""
    for r in (0.0, 10.0, 30.0, 49.0):
        sol.reset()
        u, s = sol.solve(np.array([r, 0.0, 0.0, 18.0]), np.array([0.0, 0.0]))
        assert np.all(np.isfinite(u))
        assert s > 0.0


def test_repeated_calls_are_identical(sol):
    x = np.array([-100.0, 20.0, 0.1, 18.0])
    u_nom = np.array([0.3, 0.1])
    sol.reset(tphi0=u_nom[1])
    a, _ = sol.solve(x, u_nom)
    sol.reset(tphi0=u_nom[1])
    b, _ = sol.solve(x, u_nom)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_row_holds_whenever_slack_is_zero(sol):
    """The invariant the whole filter exists to provide."""
    X, U = random_states(150, seed=23)
    for x, u_nom in zip(X, U):
        sol.reset()
        u, s = sol.solve(x, u_nom)
        if s < 1e-6:
            A, b = sol.obs[0].constraint(x, u)
            assert A @ u >= b - 1e-6


# ---------- the filter does something ----------

def test_head_on_forces_deceleration(sol):
    """Exactly head-on, 2g(e.d_perp) is zero, so bank has no authority and the
    only channel left is longitudinal. The QP MUST saturate it."""
    sol.reset()
    u, s = sol.solve(np.array([-80.0, 0.0, 0.0, 18.0]), np.array([0.0, 0.0]))
    assert u[0] == pytest.approx(-limits.A_MAX, abs=1e-6)
    assert s > 0.0

def test_rate_box_reaches_full_bank(sol):
    """Held at full nominal bank with the barrier inactive, the command MUST
    reach PHI_MAX. A rate box in the wrong units converges to 0.39 rad instead."""
    sol.reset(tphi0=0.0)
    x = np.array([-800.0, 300.0, 0.0, 18.0])
    u_nom = np.array([0.0, np.tan(limits.PHI_MAX)])
    for _ in range(200):
        u, _ = sol.solve(x, u_nom)
    assert u[1] == pytest.approx(np.tan(limits.PHI_MAX), rel=1e-6)