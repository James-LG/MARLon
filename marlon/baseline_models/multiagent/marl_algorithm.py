from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import MaybeCallback, GymEnv


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
        if continue1 is False:
            return False

        continue2, new_obs2, dones2 = defender_agent.perform_step(n_steps)
        if continue2 is False:
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

    while attacker_agent.num_timesteps < total_timesteps1 and defender_agent.num_timesteps < total_timesteps2:

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
