from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np

from stable_baselines3.common.type_aliases import GymEnv


class MarlonAgent(ABC):

    @property
    @abstractmethod
    def wrapper(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def env(self) -> GymEnv:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_timesteps(self) -> int:
        raise NotImplementedError

    @num_timesteps.setter
    @abstractmethod
    def num_timesteps(self, value):
        raise NotImplementedError

    @property
    @abstractmethod
    def n_rollout_steps(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def predict(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def perform_step(self, n_steps: int) -> Tuple[bool, Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def log_training(self, iteration: int):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def learn(self, total_timesteps: int, n_eval_episodes: int):
        """
        Use only for single-agent learning.
        For multi-agent learning you must use marl_algorithm functions.
        """
        raise NotImplementedError

    @abstractmethod
    def setup_learn(self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        eval_freq: int,
        n_eval_episodes: int,
        eval_log_path: Optional[str],
        reset_num_timesteps: bool,
        tb_log_name: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def update_progress(self, total_timesteps: int):
        raise NotImplementedError

    @abstractmethod
    def on_rollout_start(self):
        raise NotImplementedError

    @abstractmethod
    def on_rollout_end(self, new_obs: Any, dones: Any):
        raise NotImplementedError

    @abstractmethod
    def on_training_end(self):
        raise NotImplementedError

    @abstractmethod
    def save(self, filepath: str):
        raise NotImplementedError
