import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from aquanav.config import EnvConfig
from aquanav.gym_env import GymnasiumAUVEnv
from aquanav.train_ppo import FixedSeedEvaluation, learning_rate, make_env


def test_training_reward_scale_preserves_episode_and_raw_metrics():
    training = make_env("none", 0)
    reference = GymnasiumAUVEnv(EnvConfig(current_mode="none", observation_noise=0.005))
    try:
        obs, _ = training.reset(seed=42)
        expected_obs, _ = reference.reset(seed=42)
        np.testing.assert_array_equal(obs, expected_obs)
        total_reward = 0.0
        for _ in range(reference.core.config.max_steps):
            action = np.array([0.5, 0.0, 0.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = training.step(action)
            expected_obs, expected_reward, expected_terminated, expected_truncated, _ = reference.step(action)
            np.testing.assert_array_equal(obs, expected_obs)
            assert reward == pytest.approx(0.01 * expected_reward)
            assert (terminated, truncated) == (expected_terminated, expected_truncated)
            total_reward += expected_reward
            if terminated or truncated:
                assert info["episode"]["r"] == pytest.approx(total_reward, abs=1e-6)
                break
        else:
            pytest.fail("Episode did not terminate within the configured step budget")
    finally:
        training.close()
        reference.close()


def test_learning_rate_decay_and_rollout_overshoot():
    assert learning_rate(1.0) == pytest.approx(3e-4)
    assert learning_rate(0.5) == pytest.approx(1.515e-4)
    assert learning_rate(0.0) == pytest.approx(3e-6)
    assert learning_rate(-0.01) == pytest.approx(3e-6)
    assert learning_rate(1.01) == pytest.approx(3e-4)


@pytest.mark.parametrize("checkpoint", ("best_model", "final_model", "last_model"))
def test_checkpoint_evaluation_choices(monkeypatch, checkpoint):
    from aquanav.evaluate_ppo import get_args

    monkeypatch.setattr("sys.argv", ["evaluate_ppo", "--checkpoint", checkpoint])
    assert get_args().checkpoint == checkpoint


@pytest.mark.parametrize("success_rates", (
    [0.5, 1.0, 0.95, 0.8, 0.7],
    [1.0, 0.9, 1.0, 0.9, 0.9, 0.9],
))
def test_validation_early_stop_and_final_assessment(tmp_path, monkeypatch, success_rates):
    from aquanav import train_ppo

    class Model:
        def save(self, path):
            assert path == tmp_path / "best_model"

    callback = FixedSeedEvaluation("none", tmp_path, every=1, episodes=20, seed=20_000)
    callback.model = Model()
    stats = {key: 0.0 for key in callback.fields[1:]}
    stats["episodes"] = 20
    monkeypatch.setattr(train_ppo, "evaluate", lambda *args: [])
    monkeypatch.setattr(train_ppo, "aggregate", lambda *args: stats.copy())
    for step, success_rate in enumerate(success_rates, start=1):
        callback.num_timesteps = step
        stats["success_rate"] = success_rate
        assert callback._on_step() == (step < len(success_rates))
    assert callback.stop_requested
    assert callback.degraded_evaluations == 3
    selected_step = callback.best_step
    callback._on_training_end()
    assert callback.degraded_evaluations == 3
    assert callback.best_step == selected_step
    assert callback.best_score[0] == 1.0
