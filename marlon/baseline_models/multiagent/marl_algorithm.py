from typing import Optional

import numpy as np

from stable_baselines3.common.type_aliases import GymEnv

from marlon.baseline_models.multiagent.marlon_agent import MarlonAgent

def collect_rollouts(
    attacker_agent: MarlonAgent,
    defender_agent: MarlonAgent
) -> bool:
    """
    Collect experiences using the current policy and fill a ``RolloutBuffer``.
    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.

    :param env: The training environment
    :param callback: Callback that will be called at each step
        (and at the beginning and end of the rollout)
    :param rollout_buffer: Buffer to fill with rollouts
    :param n_steps: Number of experiences to collect per environment
    :return: True if function returned with at least `n_rollout_steps`
        collected, False if callback terminated rollout prematurely.
    """
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
):
    """
    Runs an episode with two agents until max_steps is reached or the
    environment's done flag is set.
    """
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
