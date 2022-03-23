from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np

from stable_baselines3.common.type_aliases import GymEnv

class EvaluationAgent(ABC):
    '''
    Common interface for agents that can be evaluated.
    NOTE: Does not include training-related methods. See MarlonAgent instead.
    '''

    @property
    @abstractmethod
    def wrapper(self) -> GymEnv:
        raise NotImplementedError

    @property
    @abstractmethod
    def env(self) -> GymEnv:
        '''The environment this agent will train and evaluate on.'''
        raise NotImplementedError

    @abstractmethod
    def predict(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def post_predict_callback(self, observation, reward, done, info):
        raise NotImplementedError

class MarlonAgent(EvaluationAgent):
    '''Common interface for agents used in MARL algorithms.'''

    @property
    @abstractmethod
    def num_timesteps(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_rollout_steps(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def log_interval(self) -> int:
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
        '''
        Train all agents in the universe for the specified amount of steps or episodes,
        which ever comes first.

        Use only for single-agent learning.
        For multi-agent learning you must use marl_algorithm functions.

        Parameters
        ----------
        total_timesteps : int
            The maximum number of timesteps to train for, across all episodes.
        n_eval_episodes : int
            The maximum number of episodes to train for, regardless of timesteps.
        '''
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
