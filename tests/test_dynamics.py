import numpy as np
import pytest

from aquanav.config import AUVConfig
from aquanav.currents import NoCurrent, UniformCurrent, make_current
from aquanav.dynamics import PlanarAUV


def test_zero_state_stays_zero_without_force_or_current():
    vehicle = PlanarAUV(AUVConfig(), NoCurrent())
    state = vehicle.step(np.zeros(6), np.zeros(3), time=0.0, dt=0.1)
    np.testing.assert_allclose(state, 0.0, atol=1e-12)


def test_forward_force_increases_surge_velocity():
    vehicle = PlanarAUV(AUVConfig(), NoCurrent())
    state = vehicle.step(np.zeros(6), np.array([0.5, 0.0, 0.0]), 0.0, 0.1)
    assert state[3] > 0.0


def test_two_current_modes():
    point = np.array([2.0, -3.0])
    np.testing.assert_array_equal(make_current("none", 0.55).velocity(point, 1.0), [0, 0])
    velocity = make_current("uniform", 0.55).velocity(point, 1.0)
    np.testing.assert_allclose(np.linalg.norm(velocity), 0.55)
    np.testing.assert_allclose(velocity, UniformCurrent().velocity(point, 999.0))
    with pytest.raises(ValueError):
        make_current("vortex", 0.55)
