from abc import ABC, abstractmethod
import logging
import os
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
    '''Builder capable of creating a generic MarlonAgent given a wrapper.'''

    @abstractmethod
    def build(self, wrapper: GymEnv, logger: logging.Logger) -> MarlonAgent:
        '''
        Build a generic MarlonAgent given a wrapper.

        Parameters
        ----------
            wrapper : GymEnv
                The wrapper the agent will operate on.
            logger : Logger
                Logger instance to write logs with.

        Returns
        -------
            A MarlonAgent built for the given wrapper.
        '''
        raise NotImplementedError

class MultiAgentUniverse:
    '''
    Helps build multi-agent environments by handling intracacies of various
    combinations of attacker and defender agents.
    '''

    @classmethod
    def build(cls,
        attacker_builder: AgentBuilder,
        attacker_invalid_action_reward = -1,
        defender_builder: Optional[AgentBuilder] = None,
        defender_invalid_action_reward = -1,
        env_id: str = "CyberBattleToyCtf-v0",
        max_timesteps: int = 2000,
        attacker_loss_reward: float = -5000.0,
        defender_loss_reward: float = -5000.0):
        '''
        Static factory method to create a MultiAgentUniverse with the given options.

        Parameters
        ----------
        attacker_builder : AgentBuilder
            A builder that will create an attacker MarlonAgent.
        attacker_enable_action_penalty : bool
            Whether the AttackerEnvWrapper should enable invalid action penalties.
        defender_builder : AgentBuilder
            A builder that will create a defender MarlonAgent.
        defender_enable_action_penalty : bool
            Whether the DefnederEnvWrapper should enable invalid action penalties.
        env_id : str
            The gym environment ID to create. Should return a type that inherits CyberBattleEnv.
        max_timesteps : int
            The max timesteps per episode before the simulation is forced to end.
            Useful if training gets stuck on a single episode for too long.

        Returns
        -------
            A MultiAgentUniverse configured with the given options.
        '''

        log_level = os.environ.get('LOGLEVEL', 'INFO').upper()
        logger = logging.Logger('marlon', level=log_level)
        console_handler = logging.StreamHandler()
        logger.addHandler(console_handler)

        if defender_builder:
            cyber_env = gym.make(
                env_id,
                defender_constraint=DefenderConstraint(maintain_sla=0.60),
                losing_reward = defender_loss_reward)
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
        attacker_agent = attacker_builder.build(attacker_wrapper, logger)

        defender_agent = None
        if defender_builder:
            defender_wrapper = DefenderEnvWrapper(
                cyber_env=cyber_env,
                event_source=event_source,
                attacker_reward_store=attacker_wrapper,
                max_timesteps=max_timesteps,
                invalid_action_reward=defender_invalid_action_reward,
                defender=True
            )
            defender_agent = defender_builder.build(defender_wrapper, logger)

        return MultiAgentUniverse(
            attacker_agent=attacker_agent,
            defender_agent=defender_agent,
            max_timesteps=max_timesteps,
            logger=logger
        )

    def __init__(self,
        attacker_agent: MarlonAgent,
        defender_agent: Optional[MarlonAgent],
        max_timesteps: int,
        logger: logging.Logger):

        self.attacker_agent = attacker_agent
        self.defender_agent = defender_agent
        self.max_timesteps = max_timesteps
        self.logger = logger

    def learn(self, total_timesteps: int, n_eval_episodes: int):
        '''
        Train all agents in the universe for the specified amount of steps or episodes,
        which ever comes first.

        Parameters
        ----------
        total_timesteps : int
            The maximum number of timesteps to train for, across all episodes.
        n_eval_episodes : int
            The maximum number of episodes to train for, regardless of timesteps.
        '''

        self.logger.info('Training started')
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
        self.logger.info('Training complete')

    def evaluate(self, n_episodes: int):
        '''
        Evaluate all agents in the universe for the given number of episodes.

        Parameters
        ----------
        n_episodes : int
            The number of episodes to evaluate for.
            Results will be calculated as averages per episode.
        '''
        self.logger.info('Evaluation started')

        attacker_rewards = []
        defender_rewards = []
        episode_steps = []

        for i in range(n_episodes):
            self.logger.info(f'Evaluating episode {i+1} of {n_episodes}')
            episode_rewards1, episode_rewards2, _ = marl_algorithm.run_episode(
                attacker_agent=self.attacker_agent,
                defender_agent=self.defender_agent,
                max_steps=self.max_timesteps
            )

            attacker_rewards.append(sum(episode_rewards1))
            episode_steps.append(len(episode_rewards1))

            if self.defender_agent:
                defender_rewards.append(sum(episode_rewards2))

        attacker_rewards = np.array(attacker_rewards)
        defender_rewards = np.array(defender_rewards)

        mean_length = np.mean(episode_steps)
        std_length = np.std(episode_steps)

        mean1 = np.mean(attacker_rewards)
        std_dev1 = np.std(attacker_rewards)

        self.logger.info('-----------------------')
        self.logger.info('| Evaluation Complete |')
        self.logger.info('-----------------------')
        self.logger.info('| Episode length:      |')
        self.logger.info('|   mean: %.2f', mean_length)
        self.logger.info('|   std_dev: %.2f', std_length)
        self.logger.info('-----------------------')
        self.logger.info('| Attacker:           |')
        self.logger.info('|   mean:    %.2f', mean1)
        self.logger.info('|   std_dev: %.2f', std_dev1)
        self.logger.info('-----------------------')

        if self.defender_agent:
            mean2 = np.mean(defender_rewards)
            std_dev2 = np.std(defender_rewards)

            self.logger.info('| Defender:           |')
            self.logger.info('|   mean:    %.2f', mean2)
            self.logger.info('|   std_dev: %.2f', std_dev2)
            self.logger.info('-----------------------')

    def save(self,
        attacker_filepath: Optional[str] = None,
        defender_filepath: Optional[str] = None):
        '''
        Save all agents in the universe at the specified file paths.

        It is safe to supply a file path for an agent that does not actually
        exist in the universe. Therefore it is safe for both file paths
        to always be supplied, regardless of universe configuration.

        Parameters
        attacker_filepath : Optional[str]
            The file path to save the attacker agent.
        '''

        if attacker_filepath is not None:
            self.logger.info('Attacker agent saving...')
            self.attacker_agent.save(attacker_filepath)

        if defender_filepath is not None and\
            self.defender_agent is not None:

            self.logger.info('Defender agent saving...')
            self.defender_agent.save(defender_filepath)
