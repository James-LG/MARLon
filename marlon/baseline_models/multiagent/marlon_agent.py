from abc import ABC, abstractmethod
import time
from typing import Any, Optional, Tuple

import numpy as np
import torch as th

import gym

from stable_baselines3.common.logger import Logger
from stable_baselines3.common.utils import safe_mean, obs_as_tensor
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import MaybeCallback, GymEnv
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

class MarlonAgent(ABC):
    @abstractmethod
    def perform_step(self, n_steps: int, callback: BaseCallback) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def rollout_buffer(self) -> RolloutBuffer:
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
    def num_timesteps(self, value) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_rollout_steps(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def log_training(self, iteration: int):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def setup_learn(self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        callback: MaybeCallback,
        eval_freq: int,
        n_eval_episodes: int,
        eval_log_path: Optional[str],
        reset_num_timesteps: bool,
        tb_log_name: str):
        raise NotImplementedError

    @abstractmethod
    def update_progress(self, total_timesteps: int):
        raise NotImplementedError

    @abstractmethod
    def on_rollout_start(self, callback: BaseCallback):
        raise NotImplementedError

    @abstractmethod
    def on_rollout_end(self, callback: BaseCallback, new_obs: Any, dones: Any):
        raise NotImplementedError

class BaselineMarlonAgent(MarlonAgent):
    def __init__(self, baseline_model: OnPolicyAlgorithm):
        self.baseline_model: OnPolicyAlgorithm = baseline_model
        self.callback: MaybeCallback = None

    def perform_step(self,
        n_steps: int,
        callback: BaseCallback) -> Tuple[bool, Any, Any]:
        if self.baseline_model.use_sde and self.baseline_model.sde_sample_freq > 0 and n_steps % self.baseline_model.sde_sample_freq == 0:
            # Sample a new noise matrix
            self.baseline_model.policy.reset_noise(self.env.num_envs)

        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(self.baseline_model._last_obs, self.baseline_model.device)
            actions1, values1, log_probs1 = self.baseline_model.policy.forward(obs_tensor)

        actions1 = actions1.cpu().numpy()

        # Rescale and perform action
        clipped_actions1 = actions1
        # Clip the actions to avoid out of bound error
        if isinstance(self.baseline_model.action_space, gym.spaces.Box):
            clipped_actions1 = np.clip(actions1, self.baseline_model.action_space.low, self.baseline_model.action_space.high)

        new_obs1, rewards1, dones1, infos1 = self.env.step(clipped_actions1)

        self.num_timesteps += self.env.num_envs

        # Give access to local variables
        callback.update_locals(locals())
        if callback.on_step() is False:
            return False, new_obs1, dones1

        self.baseline_model._update_info_buffer(infos1)

        if isinstance(self.baseline_model.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete action
            actions1 = actions1.reshape(-1, 1)

        self.rollout_buffer.add(self.baseline_model._last_obs, actions1, rewards1, self.baseline_model._last_episode_starts, values1, log_probs1)
        self.baseline_model._last_obs = new_obs1
        self.baseline_model._last_episode_starts = dones1

        return True, new_obs1, dones1

    @property
    def rollout_buffer(self) -> RolloutBuffer:
        return self.baseline_model.rollout_buffer

    @property
    def env(self) -> GymEnv:
        return self.baseline_model.env

    @property
    def num_timesteps(self) -> int:
        return self.baseline_model.num_timesteps

    @num_timesteps.setter
    def num_timesteps(self, value) -> int:
        """ Ensure the baseline model's num_timesteps is always up to date. """
        self.baseline_model.num_timesteps = value

    @property
    def n_rollout_steps(self) -> int:
        return self.baseline_model.n_steps

    def log_training(self, iteration: int):
        fps = int(self.baseline_model.num_timesteps / (time.time() - self.baseline_model.start_time))
        self.baseline_model.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.baseline_model.ep_info_buffer) > 0 and len(self.baseline_model.ep_info_buffer[0]) > 0:
            self.baseline_model.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.baseline_model.ep_info_buffer]))
            self.baseline_model.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.baseline_model.ep_info_buffer]))
        self.baseline_model.logger.record("time/fps", fps)
        self.baseline_model.logger.record("time/time_elapsed", int(time.time() - self.baseline_model.start_time), exclude="tensorboard")
        self.baseline_model.logger.record("time/total_timesteps", self.baseline_model.num_timesteps, exclude="tensorboard")
        self.baseline_model.logger.dump(step=self.baseline_model.num_timesteps)

    def train(self):
        self.baseline_model.train()

    def setup_learn(self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        callback: MaybeCallback,
        eval_freq: int,
        n_eval_episodes: int,
        eval_log_path: Optional[str],
        reset_num_timesteps: bool,
        tb_log_name: str) -> Tuple[int, BaseCallback]:

        return self.baseline_model._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name
        )

    def update_progress(self, total_timesteps: int):
        return self.baseline_model._update_current_progress_remaining(self.num_timesteps, total_timesteps)

    def on_rollout_start(self, callback: BaseCallback):
        assert self.baseline_model._last_obs is not None, "No previous observation was provided"

        # Switch to eval mode (this affects batch norm / dropout)
        self.baseline_model.policy.set_training_mode(False)

        self.rollout_buffer.reset()

        # Sample new weights for the state dependent exploration
        if self.baseline_model.use_sde:
            self.baseline_model.policy.reset_noise(self.env.num_envs)

        callback.on_rollout_start()

    def on_rollout_end(self, callback: BaseCallback, new_obs: Any, dones: Any):
        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = obs_as_tensor(new_obs, self.baseline_model.device)
            _, values, _ = self.baseline_model.policy.forward(obs_tensor)

        self.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()