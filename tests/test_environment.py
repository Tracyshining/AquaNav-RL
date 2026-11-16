import numpy as np

from aquanav.config import EnvConfig
from aquanav.environment import AUVNavigationEnv


def test_reset_and_step_shapes():
    env = AUVNavigationEnv(EnvConfig(current_mode="none"))
    observation, info = env.reset(seed=3)
    assert observation.shape == (10,)
    assert info["state"].shape == (6,)
    next_observation, reward, terminated, truncated, info = env.step(np.zeros(3))
    assert next_observation.shape == (10,)
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated


def test_invalid_action_shape_is_rejected():
    env = AUVNavigationEnv()
    env.reset(seed=1)
    try:
        env.step(np.zeros(2))
    except ValueError:
        return
    raise AssertionError("Expected invalid action shape to raise ValueError")

