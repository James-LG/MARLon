#!/usr/bin/python3.8

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# -*- coding: utf-8 -*-
"""CLI to run the baseline Deep Q-learning and Random agents
   on a sample CyberBattle gym environment and plot the respective
   cummulative rewards in the terminal.

Example usage:

    python3.8 -m run --training_episode_count 50  --iteration_count 9000 --rewardplot_with 80  --chain_size=20 --ownership_goal 1.0

"""
import gym
import argparse
import cyberbattle._env.cyberbattle_env as cyberbattle_env

parser = argparse.ArgumentParser(description='Run simulation with DQL baseline agent.')

parser.add_argument('--reward_goal', default=2180, type=int,
                    help='minimum target rewards to reach for the attacker to reach its goal')

parser.add_argument('--ownership_goal', default=1.0, type=float,
                    help='percentage of network nodes to own for the attacker to reach its goal')

parser.add_argument('--chain_size', default=4, type=int,
                    help='size of the chain of the CyberBattleChain sample environment')

args = parser.parse_args()

env = gym.make(
    'CyberBattleChain-v0',
    size=args.chain_size,
    attacker_goal=cyberbattle_env.AttackerGoal(
        own_atleast_percent=args.ownership_goal,
        reward=args.reward_goal))

print(env.action_space)
for i_episode in range(20):
    observation = env.reset()
    print(observation)
    for t in range(100):
        action = env.sample_valid_action()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
