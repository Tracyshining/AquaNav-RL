"""Gymnasium wrapper used by vanilla Stable-Baselines3 PPO."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import AUVConfig, EnvConfig
from .environment import AUVNavigationEnv


class GymnasiumAUVEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        env_config: EnvConfig | None = None,
        auv_config: AUVConfig | None = None,
    ) -> None:
        super().__init__()
        self.core = AUVNavigationEnv(env_config, auv_config)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.core.observation_size,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Gymnasium's seeded RNG stays live across episodes. In the source
        # implementation, core.reset(seed=None) created a new unseeded RNG,
        # making training randomization non-reproducible after the first reset.
        episode_seed = int(self.np_random.integers(0, 2**32 - 1))
        randomize = bool((options or {}).get("randomize", True))
        return self.core.reset(seed=episode_seed, randomize=randomize)

    def step(self, action):
        return self.core.step(action)
