import numpy as np
import pytest
pytest.importorskip("gymnasium")

from aquanav.config import EnvConfig


def test_gymnasium_seed_and_episode_randomization():
    from aquanav.gym_env import GymnasiumAUVEnv

    a = GymnasiumAUVEnv(EnvConfig(current_mode="uniform"))
    b = GymnasiumAUVEnv(EnvConfig(current_mode="uniform"))
    for i in range(3):
        oa, _ = a.reset(seed=12 if i == 0 else None)
        ob, _ = b.reset(seed=12 if i == 0 else None)
        np.testing.assert_array_equal(oa, ob)
        np.testing.assert_array_equal(a.core.state, b.core.state)
    assert a.action_space.shape == (3,)
    assert a.observation_space.shape == (10,)
    a.close()
    b.close()
