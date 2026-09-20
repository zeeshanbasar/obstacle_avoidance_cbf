"""Day 12 -- stacked barrier rows, fixed N, per-row slack.

ASSUMED API. Everything the tests depend on is reached through `make_solver`
below, so if your signatures differ, edit that one function and nothing else.

    Solver(x_dim, u_dim, dt, max_n_obs)   QP built once, sized for max_n_obs
    sol.set_obstacles([Obstacle, ...])    len <= max_n_obs, sets sol.n_obs,
                                          pads the remaining rows
    sol.reset(tphi0)                      seeds the roll-rate window
    u, s = sol.solve(x, u_nom)            s has max_n_obs entries, in list order

STATE REGIMES. The barrier row is A @ u >= b, with

    A = [2 e.d,  2 g e.d_perp]
    b = -(2 V**2 + (K_0 + K_1) h_dot + K_0 K_1 h)

so whether a row is inactive, active-and-feasible, or active-and-infeasible is
decided by the geometry, not by the solver. With K_0 = K_1 = 1, r_o = 50 m and
V = 18 m/s the row only activates inside about 92 m of the obstacle centre.
Three states are used throughout, all with the obstacle at the origin:

    X_ACTIVE   (-80, 32)  b = 188    reachable bound 333   active, feasible
    X_HEADON   (-80,  0)  b = 1212   reachable bound 320   active, infeasible
                          bank column is 0 -- the degenerate geometry
    OBS_FAR    centre (5000, 5000)   b = -5.0e7             inactive

`test_fixture_states_are_in_the_intended_regime` re-derives all three. It MUST
be kept: a change to K_0, K_1, A_MAX or PHI_DOT_MAX can move a state out of its
regime, and several tests below would then pass without exercising anything.
"""

import numpy as np
import pytest

from src import limits
from src.filter import Solver
from src.obstacle import Obstacle

DT = 0.02
N_MAX = 4

U_NOM = np.array([0.0, 0.0])          # straight and level, inside every bound

X_ACTIVE = np.array([-80.0, 32.0, 0.0, 18.0])
X_HEADON = np.array([-80.0, 0.0, 0.0, 18.0])

SLACK_TOL = 1e-6
U_ATOL = 1e-6

def obs_near(delta=0.0):
    return Obstacle(po_x=0.0, po_y=0.0, po_r=50.0, delta=delta)


def obs_far():
    return Obstacle(po_x=5000.0, po_y=5000.0, po_r=50.0)


def make_solver(obstacles, max_n_obs=N_MAX, tphi0=0.0):
    sol = Solver(x_dim=4, u_dim=2, dt=DT, max_n_obs=max_n_obs)
    sol.set_obstacles(list(obstacles))
    sol.reset(tphi0=tphi0)
    return sol


def solve_once(obstacles, x, max_n_obs=N_MAX, u_nom=U_NOM, tphi0=0.0):
    """One solve from a clean solver. Every test starts from the same roll-rate
    window, so results are comparable across solvers."""
    u, s = make_solver(obstacles, max_n_obs, tphi0).solve(x, u_nom)
    return np.asarray(u).ravel(), np.asarray(s).ravel()


# ---------- the fixture states themselves ----------

def test_fixture_states_are_in_the_intended_regime():
    """Guard. If this fails, the states below no longer test what they claim
    and the rest of the file is vacuous -- fix the states, not this test."""
    d = limits.PHI_DOT_MAX * DT
    t_hi, t_lo = np.tan(d), np.tan(-d)       # window around tphi0 = 0

    def reachable_max(A):
        return (max(A[0] * limits.A_MAX, -A[0] * limits.A_MAX)
                + max(A[1] * t_hi, A[1] * t_lo))

    A, b = obs_near().constraint(X_ACTIVE, U_NOM)
    assert A @ U_NOM < b, "X_ACTIVE: nominal already satisfies the row"
    assert reachable_max(A) > b, "X_ACTIVE: no admissible input satisfies the row"
    assert abs(A[1]) > 1.0, "X_ACTIVE: bank column has vanished"

    A, b = obs_near().constraint(X_HEADON, U_NOM)
    assert reachable_max(A) < b, "X_HEADON: the row is satisfiable, so no slack"
    assert abs(A[1]) < 1e-6, "X_HEADON: expected the degenerate zero-offset case"

    A, b = obs_far().constraint(X_ACTIVE, U_NOM)
    assert A @ U_NOM > b, "OBS_FAR: the distant row is active"


# ---------- bookkeeping ----------

def test_n_obs_follows_the_list():
    """n_obs MUST be derived, never passed in. A caller-supplied count can
    disagree with the list and pad from the wrong row, silently dropping a
    real obstacle or leaving the previous scenario's row in the block."""
    sol = make_solver([obs_near(), obs_far()])
    assert sol.n_obs == 2
    sol.set_obstacles([obs_near()])
    assert sol.n_obs == 1
    sol.set_obstacles([])
    assert sol.n_obs == 0


def test_too_many_obstacles_is_rejected():
    """The QP is sized once. Overflow MUST raise, not truncate -- a silently
    dropped obstacle is an unfiltered obstacle."""
    sol = make_solver([], max_n_obs=2)
    with pytest.raises(Exception):
        sol.set_obstacles([obs_near(), obs_far(), obs_near()])


def test_slack_vector_is_max_n_obs_long():
    """Documents the slice the campaign runners owe: entries past n_obs are
    padding, and recording them stores fake obstacles with perfect margin."""
    _, s = solve_once([obs_near()], X_ACTIVE, max_n_obs=N_MAX)
    assert s.size == N_MAX


# ---------- padding is inert ----------

def test_empty_obstacle_list_is_pass_through():
    """With every row padded the barrier block MUST contribute nothing, so the
    certified command equals the nominal one. The strongest padding test there
    is: no real row exists that could mask a padded row that binds."""
    u, s = solve_once([], X_ACTIVE)
    np.testing.assert_allclose(u, U_NOM, atol=1e-6)
    np.testing.assert_allclose(s, 0.0, atol=SLACK_TOL)


def test_padding_does_not_change_the_solution():
    """Same obstacle, two solver sizes. The padded rows MUST NOT shift the
    optimum -- the check that A = [0, 0] and b = -1 really are inert."""
    u1, s1 = solve_once([obs_near()], X_ACTIVE, max_n_obs=1)
    u4, s4 = solve_once([obs_near()], X_ACTIVE, max_n_obs=N_MAX)
    np.testing.assert_allclose(u4, u1, atol=1e-9)
    np.testing.assert_allclose(s4[0], s1[0], atol=SLACK_TOL)
    assert not np.allclose(u1, U_NOM, atol=1e-3), "row never bound; test vacuous"


def test_padded_slacks_stay_zero_when_a_real_row_is_infeasible():
    """Under stress the solver relaxes rows. Padded rows MUST NOT be among
    them, or the recorded per-row slack names an obstacle that does not exist."""
    _, s = solve_once([obs_near()], X_HEADON, max_n_obs=N_MAX)
    assert s[0] > limits_slack_tol()
    np.testing.assert_allclose(s[1:], 0.0, atol=SLACK_TOL)


def limits_slack_tol():
    return 1e-6


# ---------- composition ----------

def test_distant_obstacle_does_not_interfere():
    """Adding a row that cannot bind MUST NOT move the solution. Catches a
    shared slack, a mis-sized parameter, and a row written to the wrong index."""
    u_one, _ = solve_once([obs_near()], X_ACTIVE)
    u_two, s_two = solve_once([obs_near(), obs_far()], X_ACTIVE)
    np.testing.assert_allclose(u_two, u_one, atol=1e-9)
    np.testing.assert_allclose(s_two, 0.0, atol=SLACK_TOL)


def test_inactive_row_keeps_its_margin():
    """The distant row MUST hold with slack to spare at the certified input.
    A row that ends up tight was built from the wrong obstacle."""
    u, _ = solve_once([obs_near(), obs_far()], X_ACTIVE)
    A, b = obs_far().constraint(X_ACTIVE, U_NOM)
    assert A @ u - b > 1e4


def test_slack_is_attributable_to_the_offending_row():
    """THE test for per-row slack. Obstacle B cannot be satisfied inside the
    input box; obstacle A is 7 km away. A's slack MUST stay at zero.

    Under a single shared slack this fails: the relaxation B needs is applied
    to every row at once, so A is relaxed by ~900 as well and the campaign
    cannot say which obstacle lost the guarantee."""
    _, s = solve_once([obs_far(), obs_near()], X_HEADON)
    assert s[1] > limits_slack_tol(), "the infeasible row did not take slack"
    assert s[0] == pytest.approx(0.0, abs=1e-9), "slack leaked onto a safe row"


def test_row_order_does_not_change_the_command():
    """Rows are a set. Permuting the list MUST permute the slack vector and
    leave the certified command untouched."""
    u_a, s_a = solve_once([obs_far(), obs_near()], X_HEADON)
    u_b, s_b = solve_once([obs_near(), obs_far()], X_HEADON)
    np.testing.assert_allclose(u_b, u_a, atol=1e-9)
    assert s_b[0] == pytest.approx(s_a[1], abs=1e-9)
    assert s_b[1] == pytest.approx(s_a[0], abs=1e-9)


def test_set_obstacles_clears_the_previous_scenario():
    """The campaign runners reuse one solver across 500 scenarios. A row left
    over from the previous scenario would filter against an obstacle that is
    no longer there -- and it would look like a plausible result."""
    sol = make_solver([obs_near(), obs_far()])
    sol.solve(X_ACTIVE, U_NOM)

    sol.set_obstacles([obs_near()])
    sol.reset(tphi0=0.0)
    u_reused, s_reused = sol.solve(X_ACTIVE, U_NOM)

    u_fresh, s_fresh = solve_once([obs_near()], X_ACTIVE)
    np.testing.assert_allclose(u_reused, u_fresh, atol=1e-9)
    np.testing.assert_allclose(s_reused, s_fresh, atol=SLACK_TOL)


# ---------- the barrier still means what it meant ----------

def test_single_row_binds_at_equality():
    """With one active feasible row the QP MUST sit exactly on the constraint:
    the barrier is a bound to be met, not a cost to be traded against. Derived
    from the geometry, so it also pins the (A, b) sign convention."""
    u, s = solve_once([obs_near()], X_ACTIVE)
    A, b = obs_near().constraint(X_ACTIVE, U_NOM)
    assert s[0] == pytest.approx(0.0, abs=SLACK_TOL)
    assert A @ u == pytest.approx(b, rel=1e-6)


def test_inflation_carries_through_the_stack():
    """delta MUST still reach the row when obstacles arrive as a list. An
    inflated obstacle demands strictly more of the input than a bare one."""
    u_bare, _ = solve_once([obs_near(delta=0.0)], X_ACTIVE)
    u_inflated, _ = solve_once([obs_near(delta=10.0)], X_ACTIVE)
    assert abs(u_inflated[0]) > abs(u_bare[0])

test_single_row_binds_at_equality()