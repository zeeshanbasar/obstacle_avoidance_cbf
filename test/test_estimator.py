import numpy as np
import pytest

from src import limits
from src.estimator import Sensor
from src.vehicle import f
from src.rk45 import simulate_with_filter
from src.obstacle import Obstacle
from src.filter import Solver
from src.guidance import nominal

DT = 0.02
SIGMA = np.array([limits.SIGMA_PX, limits.SIGMA_PY,
                  limits.SIGMA_PSI, limits.SIGMA_V])


@pytest.fixture
def x():
    return np.array([-120.0, 37.0, 0.4, 18.0])


def stream(s, x, n, dt=DT):
    """n measurements at successive timesteps. At 50 Hz every call updates."""
    return np.array([s.measure(x, k * dt) for k in range(n)])


# ---------- noise off: MUST be an exact pass-through ----------

def test_zero_noise_returns_state_exactly(x):
    """The regression path. Day 6 results MUST reproduce with a sensor in the
    loop, so this has to be exact, not approximate."""
    out = Sensor(isNoise=False).measure(x, 0.0)
    np.testing.assert_array_equal(out, x)


def test_zero_noise_does_not_advance_the_rng(x):
    """With noise off, no draw may be consumed -- otherwise inserting the
    sensor shifts the scenario stream and the campaign stops reproducing."""
    s = Sensor(isNoise=False, seed=7)
    before = s.rng.bit_generator.state
    stream(s, x, 100)
    assert s.rng.bit_generator.state == before


def test_measure_does_not_mutate_input(x):
    for flag in (False, True):
        before = x.copy()
        Sensor(isNoise=flag).measure(x, 0.0)
        np.testing.assert_array_equal(x, before)


def test_measure_returns_a_new_array(x):
    """The caller MUST NOT be able to corrupt the held estimate."""
    s = Sensor(isNoise=False)
    out = s.measure(x, 0.0)
    assert out is not x
    out[0] = 999.0
    assert x[0] != 999.0
    assert s.measure(x, DT)[0] != 999.0


# ---------- noise on: distribution ----------

def test_noise_is_unbiased_with_correct_sigma(x):
    """Sample mean MUST match the true state; sample std MUST match sigma."""
    n = 20000
    out = stream(Sensor(isNoise=True), x, n)

    # Error in units of the standard error of the mean. 4 SEM per channel
    # gives about 1 false failure in 16000 runs across the four channels.
    z = (out.mean(axis=0) - x) / (SIGMA / np.sqrt(n))
    assert np.max(np.abs(z)) < 4.0

    np.testing.assert_allclose(out.std(axis=0, ddof=1), SIGMA, rtol=0.05)


def test_channels_are_independent(x):
    """Off-diagonal correlation MUST be near zero. Catches a copy-paste that
    reuses one draw across two channels."""
    out = stream(Sensor(isNoise=True), x, 20000)
    c = np.corrcoef((out - x).T)
    assert np.max(np.abs(c - np.eye(4))) < 0.05


def test_successive_measurements_differ(x):
    s = Sensor(isNoise=True)
    assert not np.allclose(s.measure(x, 0.0), s.measure(x, DT))


def test_same_seed_gives_same_sequence(x):
    a = stream(Sensor(isNoise=True, seed=1234), x, 50)
    b = stream(Sensor(isNoise=True, seed=1234), x, 50)
    np.testing.assert_array_equal(a, b)


def test_reseed_makes_a_scenario_reproducible(x):
    """Scenario noise MUST NOT depend on how many draws earlier scenarios used."""
    s = Sensor(isNoise=True, seed=0)

    s.reseed(100)
    a = stream(s, x, 50)

    for k in range(777):                  # an unrelated scenario running
        s.measure(x, k * DT)

    s.reseed(100)
    b = stream(s, x, 50)

    np.testing.assert_array_equal(a, b)


def test_reseed_clears_the_held_estimate(x):
    """A new scenario MUST NOT inherit the previous scenario's held state."""
    s = Sensor(isNoise=False, rate_hz=5.0, seed=0)
    s.measure(x, 0.0)
    s.reseed(1)
    out = s.measure(x + 100.0, 0.2)      # mid-hold for the OLD stream
    np.testing.assert_allclose(out, x + 100.0)


# ---------- zero-order hold ----------

def test_full_rate_updates_every_call(x):
    """At rate_hz = 1/dt the hold MUST be inert. This is the no-op guard that
    keeps the 50 Hz regression comparable with Day 9."""
    s = Sensor(isNoise=True, rate_hz=1.0 / DT, seed=3)
    out = stream(s, x, 50)
    assert np.all(np.any(np.diff(out, axis=0) != 0.0, axis=1))


def test_hold_updates_only_on_the_sampling_instants():
    """At 5 Hz with dt = 0.02 the estimate MUST change on every 10th call."""
    s = Sensor(isNoise=True, rate_hz=5.0, seed=0)
    x0 = np.array([0.0, 0.0, 0.0, 18.0])
    out = np.array([s.measure(x0 + np.array([18.0 * k * DT, 0.0, 0.0, 0.0]),
                              k * DT).copy()
                    for k in range(30)])
    changed = np.flatnonzero(np.any(np.diff(out, axis=0) != 0.0, axis=1))
    np.testing.assert_array_equal(changed, [9, 19])     # updates at k = 10, 20


def test_stale_estimate_lags_the_truth():
    """The held estimate MUST be the OLD state, not the current one. Fails if
    measure falls through and returns x between updates."""
    s = Sensor(isNoise=False, rate_hz=5.0, seed=0)
    s.measure(np.array([0.0, 0.0, 0.0, 18.0]), 0.0)
    out = s.measure(np.array([1.8, 0.0, 0.0, 18.0]), 0.1)    # aircraft moved 1.8 m
    assert out[0] == pytest.approx(0.0)


def test_hold_survives_float_accumulation():
    """t is built by repeated += dt, so age never lands exactly on the period.
    An equality test on age fires rarely or never; this MUST use a threshold."""
    s = Sensor(isNoise=True, rate_hz=5.0, seed=0)
    x0 = np.array([0.0, 0.0, 0.0, 18.0])
    t = 0.0
    out = []
    for _ in range(500):                  # 10 s, 50 updates expected
        out.append(s.measure(x0, t).copy())
        t += DT
    changed = np.sum(np.any(np.diff(np.array(out), axis=0) != 0.0, axis=1))
    assert changed == 49                  # first sample at t = 0, then 49 more


def test_first_call_always_samples():
    """The aircraft MUST start with a valid estimate, whatever the rate."""
    s = Sensor(isNoise=False, rate_hz=1.0, seed=0)
    x0 = np.array([5.0, 6.0, 0.1, 18.0])
    np.testing.assert_allclose(s.measure(x0, 0.0), x0)


# ---------- estimate age ----------

def test_age_is_zero_on_update_instants():
    s = Sensor(isNoise=False, rate_hz=5.0, seed=0)
    x0 = np.array([0.0, 0.0, 0.0, 18.0])
    for k in (0, 10, 20):
        s.measure(x0, k * DT)
        assert s.age == pytest.approx(0.0, abs=1e-9)


def test_age_grows_between_updates():
    """Age MUST be the elapsed time since the last sample, set on EVERY call.
    Propagation consumes this, so a value stale by one step is a real error."""
    s = Sensor(isNoise=False, rate_hz=5.0, seed=0)
    x0 = np.array([0.0, 0.0, 0.0, 18.0])
    for k in range(10):
        s.measure(x0, k * DT)
        assert s.age == pytest.approx(k * DT, abs=1e-9)


def test_age_never_exceeds_the_update_period():
    s = Sensor(isNoise=False, rate_hz=5.0, seed=0)
    x0 = np.array([0.0, 0.0, 0.0, 18.0])
    t = 0.0
    for _ in range(500):
        s.measure(x0, t)
        assert s.age < 1.0 / 5.0 + 1e-9
        t += DT


def test_staleness_distance_matches_the_rate():
    """The quantity the delta budget is built on. Max age is one step short of
    the full period, since the sample lands on the period boundary itself."""
    V, rate = 18.0, 5.0
    s = Sensor(isNoise=False, rate_hz=rate, seed=0)
    t, worst = 0.0, 0.0
    for _ in range(200):
        x_true = np.array([V * t, 0.0, 0.0, V])
        worst = max(worst, abs(x_true[0] - s.measure(x_true, t)[0]))
        t += DT
    assert worst == pytest.approx(V * (1.0/rate - DT), rel=1e-6)


# ---------- wiring ----------

def test_one_measurement_per_timestep():
    """The controller MUST act on a single estimate per instant. Two calls give
    guidance and the filter different noise draws of the same state."""
    class CountingSensor(Sensor):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.calls = 0

        def measure(self, x, t):
            self.calls += 1
            return super().measure(x, t)

    sens = CountingSensor(isNoise=False)
    sol = Solver(x_dim=4, u_dim=2, obs=[Obstacle(0.0, 0.0, 50.0)], dt=DT)
    x0 = np.array([-800.0, 120.0, 0.0, 18.0])
    xg = np.array([800.0, 0.0, 0.0, 18.0])

    n = 200
    simulate_with_filter(f=f, sol=sol, x0=x0, xg=xg, u0=nominal, h=DT,
                         nSteps=n, sens=sens)
    assert sens.calls == n + 1


def test_filter_never_sees_the_true_state():
    """The estimate MUST be the only state the controller reads. A large fixed
    bias MUST change the commands; if it does not, the true state is leaking."""
    class BiasedSensor(Sensor):
        def measure(self, x, t):
            self.age = 0.0
            return x + np.array([200.0, 200.0, 0.0, 0.0])

    sol = Solver(x_dim=4, u_dim=2, obs=[Obstacle(0.0, 0.0, 50.0)], dt=DT)
    x0 = np.array([-300.0, 0.0, 0.0, 18.0])
    xg = np.array([600.0, 0.0, 0.0, 18.0])

    args = dict(f=f, sol=sol, x0=x0, xg=xg, u0=nominal, h=DT, nSteps=400)
    _, traj_a, _, _, _, _, _ = simulate_with_filter(sens=Sensor(isNoise=False), **args)
    _, traj_b, _, _, _, _, _ = simulate_with_filter(sens=BiasedSensor(), **args)
    assert np.linalg.norm(traj_a[-1, :2] - traj_b[-1, :2]) > 10.0

# ---------- dropout ----------

def test_dropout_extends_the_age():
    """A dropped sample MUST leave the held estimate and the clock untouched."""
    s = Sensor(isNoise=False, rate_hz=5.0, p_drop=1.0, isDrop=True)
    s.measure(np.array([0.0, 0.0, 0.0, 18.0]), 0.0)     # first call always samples
    out = s.measure(np.array([3.6, 0.0, 0.0, 18.0]), 0.2)
    assert out[0] == pytest.approx(0.0)                  # still the old estimate
    assert s.age == pytest.approx(0.2)                   # and it has aged


def test_noise_stream_is_independent_of_dropout_rate():
    """The delta and dropout sweeps MUST see the same disturbance. Only which
    samples are USED may change with p_drop."""
    x = np.array([0.0, 0.0, 0.0, 18.0])
    def run(p):
        s = Sensor(isNoise=True, rate_hz=5.0, seed=7, p_drop=p, isDrop=True)
        return np.array([s.measure(x, k * 0.02) for k in range(200)])
    a, b = run(0.0), run(0.5)
    used_a = np.unique(a, axis=0)
    used_b = np.unique(b, axis=0)
    assert len(used_b) < len(used_a)                     # fewer distinct estimates
    assert all(any(np.allclose(r, q) for q in used_a) for r in used_b)  # subset

# test_one_measurement_per_timestep()