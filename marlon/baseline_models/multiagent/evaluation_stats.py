
from typing import List, Optional
import logging

import numpy as np


class EvalutionStats:
    '''
    Represents the results from an evaluation.
    Compiles some additional statistics for ease of use.
    '''
    def __init__(self,
        episode_steps: List[float],
        attacker_rewards: List[float],
        attacker_valid_actions: List[float],
        attacker_invalid_actions: List[float],
        defender_rewards: Optional[List[float]],
        defender_valid_actions: Optional[List[float]],
        defender_invalid_actions: Optional[List[float]]) -> None:

        self.episode_steps = np.array(episode_steps)

        self.mean_length = np.mean(self.episode_steps)
        self.std_length = np.std(self.episode_steps)

        self.attacker_rewards = np.array(attacker_rewards)
        self.attacker_valid_actions = np.array(attacker_valid_actions)
        self.attacker_invalid_actions = np.array(attacker_invalid_actions)

        self.mean_attacker_reward = np.mean(self.attacker_rewards)
        self.std_attacker_reward = np.std(self.attacker_rewards)

        self.mean_attacker_valid = np.mean(self.attacker_valid_actions)
        self.std_attacker_valid = np.std(self.attacker_valid_actions)
        self.mean_attacker_invalid = np.mean(self.attacker_invalid_actions)
        self.std_attacker_invalid = np.std(self.attacker_invalid_actions)

        if defender_rewards is not None and len(defender_rewards) > 0:
            self.defender_rewards = np.array(defender_rewards)
            self.defender_valid_actions = np.array(defender_valid_actions)
            self.defender_invalid_actions = np.array(defender_invalid_actions)

            self.mean_defender_reward = np.mean(self.defender_rewards)
            self.std_defender_reward = np.std(self.defender_rewards)

            self.mean_defender_valid = np.mean(self.defender_valid_actions)
            self.std_defender_valid = np.std(self.defender_valid_actions)
            self.mean_defender_invalid = np.mean(self.defender_invalid_actions)
            self.std_defender_invalid = np.std(self.defender_invalid_actions)
        else:
            self.defender_rewards = None
            self.defender_valid_actions = None
            self.defender_invalid_actions = None

            self.mean_defender_reward = None
            self.std_defender_reward = None

            self.mean_defender_valid = None
            self.std_defender_valid = None
            self.mean_defender_invalid = None
            self.std_defender_invalid = None

    def log_results(self, logger: logging.Logger) -> None:
        logger.info('-------------------------')
        logger.info('|  Evaluation Results  |')
        logger.info('-------------------------')
        logger.info('| Episode length:       |')
        logger.info('|   mean: %.2f', self.mean_length)
        logger.info('|   std_dev: %.2f', self.std_length)
        logger.info('-------------------------')
        logger.info('| Attacker:             |')
        logger.info('|   mean:    %.2f', self.mean_attacker_reward)
        logger.info('|   std_dev: %.2f', self.std_attacker_reward)
        logger.info('|                       |')
        logger.info('|   Valid Actions:      |')
        logger.info('|      mean: %.2f', self.mean_attacker_valid)
        logger.info('|      std_dev: %.2f', self.std_attacker_valid)
        logger.info('|                       |')
        logger.info('|   Invalid Actions:    |')
        logger.info('|      mean: %.2f', self.mean_attacker_invalid)
        logger.info('|      std_dev: %.2f', self.std_attacker_invalid)
        logger.info('-------------------------')

        if self.defender_rewards is not None:
            logger.info('| Defender:             |')
            logger.info('|   mean:    %.2f', self.mean_defender_reward)
            logger.info('|   std_dev: %.2f', self.std_defender_reward)
            logger.info('|                       |')
            logger.info('|   Valid Actions:      |')
            logger.info('|      mean: %.2f', self.mean_defender_valid)
            logger.info('|      std_dev: %.2f', self.std_defender_valid)
            logger.info('|                       |')
            logger.info('|   Invalid Actions:    |')
            logger.info('|      mean: %.2f', self.mean_defender_invalid)
            logger.info('|      std_dev: %.2f', self.std_defender_invalid)
            logger.info('-------------------------')
