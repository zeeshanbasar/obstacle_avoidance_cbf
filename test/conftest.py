import numpy as np
import pytest
from src import limits

def const(u):
    """Adapt a fixed input array to the controller signature (x, xg) -> u."""
    u = np.asarray(u, dtype=float)
    return lambda x, xg: u


# Far enough that simulate's capture break never fires in open-loop tests.
DUMMY_GOAL = np.array([1.0e6, 1.0e6, 0.0, limits.V_CRUISE])


@pytest.fixture
def dt():
    return 0.02

@pytest.fixture
def trim_state():
    """Straight and level at cruise, heading +x."""
    return np.array([0.0, 0.0, 0.0, limits.V_CRUISE])

@pytest.fixture
def zero_input():
    """[a, tan(phi)] — the QP decision variables, not phi."""
    return np.array([0.0, 0.0])
