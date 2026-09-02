import numpy as np

from aquanav.config import AUVConfig
from aquanav.currents import NoCurrent, UniformCurrent, VortexCurrent
from aquanav.dynamics import PlanarAUV


def test_zero_state_stays_zero_without_force_or_current():
    vehicle = PlanarAUV(AUVConfig(), NoCurrent())
    state = vehicle.step(np.zeros(6), np.zeros(3), time=0.0, dt=0.1)
    np.testing.assert_allclose(state, 0.0, atol=1e-12)


def test_forward_force_increases_surge_velocity():
    vehicle = PlanarAUV(AUVConfig(), NoCurrent())
    state = vehicle.step(np.zeros(6), np.array([0.5, 0.0, 0.0]), 0.0, 0.1)
    assert state[3] > 0.0


def test_current_fields_are_finite():
    fields = [UniformCurrent(), VortexCurrent()]
    for field in fields:
        velocity = field.velocity(np.array([2.0, -3.0]), time=1.0)
        assert velocity.shape == (2,)
        assert np.all(np.isfinite(velocity))

