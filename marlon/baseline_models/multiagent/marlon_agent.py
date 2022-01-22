from abc import abstractmethod

import numpy as np
import torch as th

import gym

from stable_baselines3.common.utils import safe_mean, obs_as_tensor
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm


class MarlonAgent:
    @abstractmethod
    def decide_action(self):
        raise NotImplementedError


class BaselineMarlonAgent(MarlonAgent):
    def __init__(self, baseline_model: OnPolicyAlgorithm):
        self.baseline_model = baseline_model

    def decide_action(self):
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

        return clipped_actions1, values1, log_probs1