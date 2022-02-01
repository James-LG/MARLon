from typing import Any, Optional, Tuple

from stable_baselines3.common.type_aliases import MaybeCallback, GymEnv
from stable_baselines3.common.callbacks import BaseCallback

from marlon.baseline_models.multiagent.marlon_agent import MarlonAgent

class RandomMarlonAgent(MarlonAgent):
    def __init__(self,
        env: GymEnv,
        num_timesteps: int,
        n_rollout_steps: int):

        self._env = env
        self._num_timesteps = num_timesteps
        self._n_rollout_steps = n_rollout_steps

    @property
    def env(self) -> GymEnv:
        return self._env

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @num_timesteps.setter
    def num_timesteps(self, value):
        self._num_timesteps = value

    @property
    def n_rollout_steps(self) -> int:
        return self._n_rollout_steps

    def perform_step(self, n_steps: int) -> Tuple[bool, Any, Any]:
        # Choose a random action and perform the step.
        action = self.env.action_space.sample()
        _observation, _reward, done, _info = self.env.step(action)
        
        # First value is whether to continue, this is normally determined by a callback
        # which we don't have, so we'll rely on the done flag.
        continue_training = not done
        
        return continue_training, None, None

    def log_training(self, iteration: int):
        # We don't do that here.
        pass

    def train(self):
        # We don't do that here.
        pass

    def setup_learn(self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        eval_freq: int,
        n_eval_episodes: int,
        eval_log_path: Optional[str],
        reset_num_timesteps: bool,
        tb_log_name: str) -> int:
        
        # We don't do that here.
        return total_timesteps

    def update_progress(self, total_timesteps: int):
        # We don't do that here.
        pass

    def on_rollout_start(self):
        # We don't do that here.
        pass

    def on_rollout_end(self, new_obs: Any, dones: Any):
        # We don't do that here.
        pass

    def on_training_end(self):
        # We don't do that here.
        pass