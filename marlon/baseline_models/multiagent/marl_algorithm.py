'''
Functions to train an attacker and defender agents simultaneously.

Functions in this module are taken from stable-baselines3 and modified to allow multi-agent learning.
https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/on_policy_algorithm.py
'''

from typing import List, Optional, Tuple

import numpy as np

from stable_baselines3.common.type_aliases import GymEnv

from marlon.baseline_models.multiagent.marlon_agent import MarlonAgent

def collect_rollouts(
    attacker_agent: MarlonAgent,
    defender_agent: MarlonAgent
) -> bool:
    '''
    Collect experiences using the current policy and fill a ``RolloutBuffer``.
    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.

    Parameters
    ----------
    attacker_agent : MarlonAgent
        The attacker agent to train.
    defender_agent : MarlonAgent
        The defender agent to train.

    Returns
    -------
        True if function returned with at least `n_rollout_steps`
        collected, False if callback terminated rollout prematurely.
    '''
    attacker_agent.on_rollout_start()
    defender_agent.on_rollout_start()

    n_steps = 0

    while n_steps < attacker_agent.n_rollout_steps and n_steps < defender_agent.n_rollout_steps:
        continue1, new_obs1, dones1 = attacker_agent.perform_step(n_steps)
        continue2, new_obs2, dones2 = defender_agent.perform_step(n_steps)
        if continue1 is False or continue2 is False:
            return False

        n_steps += 1

    attacker_agent.on_rollout_end(new_obs1, dones1)
    defender_agent.on_rollout_end(new_obs2, dones2)

    return True

def learn(
    attacker_agent: MarlonAgent,
    defender_agent: MarlonAgent,
    total_timesteps: int,
    log_interval: int = 1,
    eval_env: Optional[GymEnv] = None,
    eval_freq: int = -1,
    n_eval_episodes: int = 5,
    tb_log_name: str = "OnPolicyAlgorithm",
    eval_log_path: Optional[str] = None,
    reset_num_timesteps: bool = True,
):
    '''
    Train an attacker and defender agent in a multi-agent scenario.

    Parameters
    ----------
    attacker_agent : MarlonAgent
        The attacker agent to train.
    defender_agent : MarlonAgent
        The defender agent to train.
    total_timesteps : int
        The total number of samples (env steps) to train on.
    log_interval : int
        The number of timesteps before logging.
    eval_env : Optional[GymEnv]
        Environment that will be used to evaluate the agent.
    eval_freq : int
        Evaluate the agent every ``eval_freq`` timesteps (this may vary a little).
    n_eval_episodes : int
        Number of episode to evaluate the agent.
    tb_log_name : str
        The name of the run for TensorBoard logging.
    eval_log_path : Optional[str]
        Path to a folder where the evaluations will be saved.
    reset_num_timesteps : bool
        Whether or not to reset the current timestep number (used in logging).
    '''

    iteration = 0

    total_timesteps1 = attacker_agent.setup_learn(
        total_timesteps, eval_env, eval_freq, n_eval_episodes,
        eval_log_path, reset_num_timesteps, tb_log_name
    )

    total_timesteps2 = defender_agent.setup_learn(
        total_timesteps, eval_env, eval_freq, n_eval_episodes,
        eval_log_path, reset_num_timesteps, tb_log_name
    )

    while attacker_agent.num_timesteps < total_timesteps1 and \
        defender_agent.num_timesteps < total_timesteps2:

        continue_training = collect_rollouts(
            attacker_agent=attacker_agent,
            defender_agent=defender_agent
        )

        if continue_training is False:
            break

        iteration += 1
        attacker_agent.update_progress(total_timesteps)
        defender_agent.update_progress(total_timesteps)

        # Display training infos
        if log_interval is not None and iteration % log_interval == 0:
            attacker_agent.log_training(iteration)
            defender_agent.log_training(iteration)

        attacker_agent.train()
        defender_agent.train()

    attacker_agent.on_training_end()
    defender_agent.on_training_end()

def run_episode(
    attacker_agent: MarlonAgent,
    defender_agent: Optional[MarlonAgent],
    max_steps: int
) -> Tuple[List[float], List[float]]:
    '''
    Runs an episode with two agents until max_steps is reached or the
    environment's done flag is set.

    Parameters
    ----------
    attacker_agent : MarlonAgent
        The attacker agent used to select offensive actions.
    defender_agent : MarlonAgent
        The defender agent used to select defensive actions.
    max_steps : int
        The max time steps before the episode is terminated.

    Returns
    -------
    attacker_rewards : List[float]
        The list of rewards at each time step for the attacker agent.
    defender_rewards : List[float]
        The list of rewards at each time step for the defender agent.
    '''
    obs1 = attacker_agent.env.reset()

    if defender_agent:
        defender_agent.wrapper.on_reset()
        obs2 = defender_agent.env.reset()

    attacker_rewards = []
    defender_rewards = []

    n_steps = 0
    while n_steps < max_steps:
        action1 = attacker_agent.predict(observation=obs1)
        obs1, rewards1, dones1, _ = attacker_agent.env.step(action1)
        if isinstance(rewards1, np.ndarray):
            rewards1 = rewards1[0]

        attacker_rewards.append(rewards1)

        if defender_agent:
            action2 = defender_agent.predict(observation=obs2)
            obs2, rewards2, dones2, _ = defender_agent.env.step(action2)
            if isinstance(rewards2, np.ndarray):
                rewards2 = rewards2[0]

            defender_rewards.append(rewards2)

            if dones1 or dones2:
                break
        else:
            if dones1:
                break
        n_steps += 1

    return attacker_rewards, defender_rewards
