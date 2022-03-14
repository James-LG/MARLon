from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import gym

from stable_baselines3.common.type_aliases import GymEnv

from cyberbattle._env.cyberbattle_env import DefenderConstraint
from marlon.baseline_models.env_wrappers.environment_event_source import EnvironmentEventSource

from marlon.baseline_models.env_wrappers.attack_wrapper import AttackerEnvWrapper
from marlon.baseline_models.env_wrappers.defend_wrapper import DefenderEnvWrapper
from marlon.baseline_models.multiagent.marlon_agent import MarlonAgent
from marlon.baseline_models.multiagent import marl_algorithm


class AgentBuilder(ABC):
    @abstractmethod
    def build(self, wrapper: GymEnv) -> MarlonAgent:
        raise NotImplementedError

class MultiAgentUniverse:
    @classmethod
    def build(cls,
        attacker_builder: AgentBuilder,
        attacker_invalid_action_reward = -1,
        defender_builder: Optional[AgentBuilder] = None,
        defender_invalid_action_reward = -1,
        env_id: str = "CyberBattleToyCtf-v0",
        max_timesteps: int = 2000,
        attacker_loss_reward: float = -5000.0):

        if defender_builder:
            cyber_env = gym.make(
                env_id,
                defender_constraint=DefenderConstraint(maintain_sla=0.80))
        else:
            cyber_env = gym.make(env_id)

        event_source = EnvironmentEventSource()

        attacker_wrapper = AttackerEnvWrapper(
            cyber_env=cyber_env,
            event_source=event_source,
            max_timesteps=max_timesteps,
            invalid_action_reward=attacker_invalid_action_reward,
            loss_reward=attacker_loss_reward
        )
        attacker_agent = attacker_builder.build(attacker_wrapper)

        defender_agent = None
        if defender_builder:
            defender_wrapper = DefenderEnvWrapper(
                cyber_env=cyber_env,
                event_source=event_source,
                attacker_reward_store=attacker_wrapper,
                max_timesteps=max_timesteps,
                enable_action_penalty=defender_invalid_action_reward,
                defender=True
            )
            defender_agent = defender_builder.build(defender_wrapper)

        return MultiAgentUniverse(
            attacker_agent=attacker_agent,
            defender_agent=defender_agent,
            max_timesteps=max_timesteps
        )

    def __init__(self,
        attacker_agent: MarlonAgent,
        defender_agent: Optional[MarlonAgent],
        max_timesteps: int):

        self.attacker_agent = attacker_agent
        self.defender_agent = defender_agent
        self.max_timesteps = max_timesteps

    def learn(self, total_timesteps: int, n_eval_episodes: int):
        if self.defender_agent:
            marl_algorithm.learn(
                attacker_agent=self.attacker_agent,
                defender_agent=self.defender_agent,
                total_timesteps=total_timesteps,
                n_eval_episodes=n_eval_episodes
            )
        else:
            self.attacker_agent.learn(
                total_timesteps=total_timesteps,
                n_eval_episodes=n_eval_episodes
            )

    def evaluate(self, n_episodes):
        attacker_rewards = []
        defender_rewards = []

        for _ in range(n_episodes):
            episode_rewards1, episode_rewards2 = marl_algorithm.run_episode(
                attacker_agent=self.attacker_agent,
                defender_agent=self.defender_agent,
                max_steps=self.max_timesteps
            )

            attacker_rewards += episode_rewards1

            if self.defender_agent:
                defender_rewards += episode_rewards2

        attacker_rewards = np.array(attacker_rewards)
        defender_rewards = np.array(defender_rewards)

        mean1 = np.mean(attacker_rewards)
        std_dev1 = np.std(attacker_rewards)

        print( '-----------------------')
        print( '| Evaluation Complete |')
        print( '-----------------------')
        print( '| Attacker:           |')
        print(f'|   mean:    {mean1:.2f}')
        print(f'|   std_dev: {std_dev1:.2f}')
        print( '-----------------------')

        if self.defender_agent:
            mean2 = np.mean(defender_rewards)
            std_dev2 = np.std(defender_rewards)

            print( '| Defender:           |')
            print(f'|   mean:    {mean2:.2f}')
            print(f'|   std_dev: {std_dev2:.2f}')
            print( '-----------------------')

    def save(self,
        attacker_filepath: Optional[str] = None,
        defender_filepath: Optional[str] = None):

        if attacker_filepath is not None:
            self.attacker_agent.save(attacker_filepath)

        if defender_filepath is not None and\
            self.defender_agent is not None:

            self.defender_agent.save(defender_filepath)
