import time
from typing import Optional, Tuple

import numpy as np
import torch as th
import gym

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import MaybeCallback, GymEnv
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import safe_mean, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

def collect_rollouts(
    attacker_model: OnPolicyAlgorithm,
    defender_model: OnPolicyAlgorithm,
    attacker_env: VecEnv,
    defender_env: VecEnv,
    attacker_callback: BaseCallback,
    defender_callback: BaseCallback,
    attacker_rollout_buffer: RolloutBuffer,
    defender_rollout_buffer: RolloutBuffer,
    n_rollout_steps: int,
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
    assert attacker_model._last_obs is not None, "No previous observation was provided for attacker"
    assert defender_model._last_obs is not None, "No previous observation was provided for defender"
    
    # Switch to eval mode (this affects batch norm / dropout)
    attacker_model.policy.set_training_mode(False)
    defender_model.policy.set_training_mode(False)

    n_steps = 0
    attacker_rollout_buffer.reset()
    defender_rollout_buffer.reset()
    # Sample new weights for the state dependent exploration
    if attacker_model.use_sde:
        attacker_model.policy.reset_noise(attacker_env.num_envs)

    if defender_model.use_sde:
        defender_model.policy.reset_noise(defender_env.num_envs)

    attacker_callback.on_rollout_start()
    defender_callback.on_rollout_start()

    while n_steps < n_rollout_steps:
        if attacker_model.use_sde and attacker_model.sde_sample_freq > 0 and n_steps % attacker_model.sde_sample_freq == 0:
            # Sample a new noise matrix
            attacker_model.policy.reset_noise(attacker_env.num_envs)

        if defender_model.use_sde and defender_model.sde_sample_freq > 0 and n_steps % defender_model.sde_sample_freq == 0:
            # Sample a new noise matrix
            defender_model.policy.reset_noise(defender_env.num_envs)

        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(attacker_model._last_obs, attacker_model.device)
            actions1, values1, log_probs1 = attacker_model.policy.forward(obs_tensor)

            obs_tensor = obs_as_tensor(defender_model._last_obs, defender_model.device)
            actions2, values2, log_probs2 = defender_model.policy.forward(obs_tensor)

        actions1 = actions1.cpu().numpy()
        actions2 = actions2.cpu().numpy()

        # Rescale and perform action
        clipped_actions1 = actions1
        # Clip the actions to avoid out of bound error
        if isinstance(attacker_model.action_space, gym.spaces.Box):
            clipped_actions1 = np.clip(actions1, attacker_model.action_space.low, attacker_model.action_space.high)
        
        new_obs1, rewards1, dones1, infos1 = attacker_env.step(clipped_actions1)

        clipped_actions2 = actions2
        if isinstance(defender_model.action_space, gym.spaces.Box):
            clipped_actions2 = np.clip(actions2, defender_model.action_space.low, defender_model.action_space.high)

        new_obs2, rewards2, dones2, infos2 = defender_env.step(clipped_actions2)

        attacker_model.num_timesteps += attacker_env.num_envs
        defender_model.num_timesteps += defender_env.num_envs

        # Give access to local variables
        attacker_callback.update_locals(locals())
        if attacker_callback.on_step() is False:
            return False

        defender_callback.update_locals(locals())
        if defender_callback.on_step() is False:
            return False

        attacker_model._update_info_buffer(infos1)
        defender_model._update_info_buffer(infos2)
        n_steps += 1

        if isinstance(attacker_model.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete action
            actions1 = actions1.reshape(-1, 1)

        attacker_rollout_buffer.add(attacker_model._last_obs, actions1, rewards1, attacker_model._last_episode_starts, values1, log_probs1)
        attacker_model._last_obs = new_obs1
        attacker_model._last_episode_starts = dones1

        if isinstance(defender_model.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete action
            actions2 = actions2.reshape(-1, 1)

        defender_rollout_buffer.add(defender_model._last_obs, actions2, rewards2, defender_model._last_episode_starts, values2, log_probs2)
        defender_model._last_obs = new_obs2
        defender_model._last_episode_starts = dones2

    with th.no_grad():
        # Compute value for the last timestep
        obs_tensor = obs_as_tensor(new_obs1, attacker_model.device)
        _, values, _ = attacker_model.policy.forward(obs_tensor)

    attacker_rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones1)
    attacker_callback.on_rollout_end()

    with th.no_grad():
        # Compute value for the last timestep
        obs_tensor = obs_as_tensor(new_obs2, defender_model.device)
        _, values, _ = defender_model.policy.forward(obs_tensor)

    defender_rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones2)
    defender_callback.on_rollout_end()

    return True

def learn(
    attacker_model: OnPolicyAlgorithm,
    defender_model: OnPolicyAlgorithm,
    total_timesteps: int,
    callback: MaybeCallback = None,
    log_interval: int = 1,
    eval_env: Optional[GymEnv] = None,
    eval_freq: int = -1,
    n_eval_episodes: int = 5,
    tb_log_name: str = "OnPolicyAlgorithm",
    eval_log_path: Optional[str] = None,
    reset_num_timesteps: bool = True,
) -> Tuple[OnPolicyAlgorithm, OnPolicyAlgorithm]:
    iteration = 0

    total_timesteps1, callback1 = attacker_model._setup_learn(
        total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
    )

    total_timesteps2, callback2 = defender_model._setup_learn(
        total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
    )

    callback1.on_training_start(locals(), globals())
    callback2.on_training_start(locals(), globals())

    while attacker_model.num_timesteps < total_timesteps1:

        continue_training = collect_rollouts(
            attacker_model=attacker_model,
            defender_model=defender_model,
            attacker_env=attacker_model.env,
            defender_env=defender_model.env,
            attacker_callback=callback1,
            defender_callback=callback2,
            attacker_rollout_buffer=attacker_model.rollout_buffer,
            defender_rollout_buffer=defender_model.rollout_buffer,
            n_rollout_steps=attacker_model.n_steps
        )

        if continue_training is False:
            break

        iteration += 1
        attacker_model._update_current_progress_remaining(attacker_model.num_timesteps, total_timesteps)
        defender_model._update_current_progress_remaining(defender_model.num_timesteps, total_timesteps)

        # Display training infos
        if log_interval is not None and iteration % log_interval == 0:
            fps = int(attacker_model.num_timesteps / (time.time() - attacker_model.start_time))
            attacker_model.logger.record("time/iterations", iteration, exclude="tensorboard")
            if len(attacker_model.ep_info_buffer) > 0 and len(attacker_model.ep_info_buffer[0]) > 0:
                attacker_model.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in attacker_model.ep_info_buffer]))
                attacker_model.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in attacker_model.ep_info_buffer]))
            attacker_model.logger.record("time/fps", fps)
            attacker_model.logger.record("time/time_elapsed", int(time.time() - attacker_model.start_time), exclude="tensorboard")
            attacker_model.logger.record("time/total_timesteps", attacker_model.num_timesteps, exclude="tensorboard")
            attacker_model.logger.dump(step=attacker_model.num_timesteps)

        attacker_model.train()
        defender_model.train()

    callback1.on_training_end()
    callback2.on_training_end()

    return attacker_model, defender_model
