from abc import abstractmethod
from enum import Enum
from typing import Optional
import gym

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from cyberbattle._env.cyberbattle_env import CyberBattleEnv, DefenderConstraint

from marlon.defender_agents.defender import PrototypeLearningDefender
from marlon.baseline_models.env_wrappers.attack_wrapper import AttackerEnvWrapper
from marlon.baseline_models.env_wrappers.defend_wrapper import DefenderEnvWrapper

class AttackerBuilder:
    @abstractmethod
    def build(self) -> AttackerEnvWrapper:
        raise NotImplementedError


class DefenderBuilder:
    @abstractmethod
    def build(self) -> DefenderEnvWrapper:
        raise NotImplementedError

class MultiAgentUniverse:
    @classmethod
    def build(cls,
        attacker_builder: AttackerBuilder,
        defender_builder: Optional[DefenderBuilder],
        env_id: str = "CyberBattleToyCtf-v0",
        max_timesteps: int = 2000):

        if defender_builder:
            cyber_env = gym.make(
                env_id,
                defender_constraint=DefenderConstraint(maintain_sla=0.80),
                defender_agent=PrototypeLearningDefender())
        else:
            cyber_env = gym.make(env_id)

        return MultiAgentUniverse(
            cyber_env=cyber_env,
            attacker_model=attacker_builder.build(),
            defender_model=defender_builder.build(),
            max_timesteps=max_timesteps
        )

    def __init__(self,
        cyber_env: CyberBattleEnv,
        attacker_model: Optional[OnPolicyAlgorithm],
        defender_model: Optional[OnPolicyAlgorithm],
        max_timesteps: int):
        
        self.cyber_env = cyber_env
        self.attacker_env = AttackerEnvWrapper(
            cyber_env=cyber_env,
            max_timesteps=max_timesteps
        )

        self.defenderr_env = DefenderEnvWrapper(
            cyber_env=cyber_env,
            attacker_reward_store=self.attacker_env,
            max_timesteps=max_timesteps,
        )

    def train(self):
        pass

    def save(self):
        pass
