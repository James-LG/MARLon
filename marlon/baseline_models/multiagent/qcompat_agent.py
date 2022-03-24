import logging
from typing import Any, Optional, Tuple

import numpy as np

from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.monitor import Monitor

from cyberbattle.agents.baseline.agent_wrapper import ActionTrackingStateAugmentation, AgentWrapper
from cyberbattle._env.cyberbattle_env import CyberBattleEnv, EnvironmentBounds
import torch
from marlon.baseline_models.env_wrappers.attack_wrapper import AttackerEnvWrapper

from marlon.baseline_models.multiagent.marlon_agent import EvaluationAgent, MarlonAgent
from marlon.baseline_models.multiagent.multiagent_universe import AgentBuilder

class QCompatibilityAgentBuilder(AgentBuilder):
    '''Assists in building RandomMarlonAgents.'''

    def __init__(self,
        file_path,
        maximum_node_count: int = 12,
        maximum_total_credentials: int = 10) -> None:
        # TODO: There must be a better way of getting the max parameters above.
        #       Extracting it straight out of the cyber env maybe?

        self.file_path = file_path
        self.maximum_node_count = maximum_node_count
        self.maximum_total_credentials = maximum_total_credentials


    def build(self, wrapper: GymEnv, logger: logging.Logger) -> MarlonAgent:
        cyber_env = wrapper.cyber_env

        environment_properties = EnvironmentBounds.of_identifiers(
            maximum_node_count=self.maximum_node_count,
            maximum_total_credentials=self.maximum_total_credentials,
            identifiers=cyber_env.identifiers
        )

        agent_env = AgentWrapper(
            cyber_env,
            ActionTrackingStateAugmentation(environment_properties, cyber_env.reset()))

        learner = torch.load(self.file_path)
        return QCompatibilityAgent(
            attacker_wrapper=wrapper,
            agent_wrapper=agent_env,
            learner=learner
        )

class QCompatibilityAgent(EvaluationAgent):
    '''
    Agent that uses CyberBattleSim's built-in Tabluar-Q and Deep-Q models.
    NOTE 1: Multi-agent learning is not supported, use only for evaluation.
    NOTE 2: Only supports attacker agents, no defender support.
    '''

    def __init__(self,
        attacker_wrapper: AttackerEnvWrapper,
        agent_wrapper: AgentWrapper,
        learner):

        self.attacker_wrapper = attacker_wrapper
        self.agent_wrapper = agent_wrapper
        self.learner = learner
        self._action_metadata = None

    @property
    def wrapper(self) -> GymEnv:
        return self.attacker_wrapper

    @property
    def env(self) -> GymEnv:
        return self.agent_wrapper

    def predict(self, observation: np.ndarray) -> np.ndarray:
        _, gym_action, self._action_metadata = self.learner.exploit(self.agent_wrapper, observation)
        if not gym_action:
            _, gym_action, self._action_metadata = self.learner.explore(self.agent_wrapper)

        return gym_action

    def post_predict_callback(self, observation, reward, done, info):
        self.learner.on_step(self.agent_wrapper, observation, reward, done, info, self._action_metadata)
