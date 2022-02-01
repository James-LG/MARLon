import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from cyberbattle._env.cyberbattle_env import DefenderConstraint
from marlon.baseline_models.env_wrappers.attack_wrapper import AttackerEnvWrapper
from marlon.baseline_models.env_wrappers.wrapper_coordinator import WrapperCoordinator
from marlon.baseline_models.multiagent.baseline_marlon_agent import BaselineMarlonAgent
from marlon.baseline_models.multiagent.marl_algorithm import learn
from marlon.baseline_models.multiagent.random_marlon_agent import RandomMarlonAgent

from marlon.defender_agents.defender import PrototypeLearningDefender
from marlon.baseline_models.env_wrappers.defend_wrapper import DefenderEnvWrapper
from marlon.baseline_models.ppo.eval_defender import evaluate

ENV_MAX_TIMESTEPS = 2000
LEARN_TIMESTEPS = 10_000
LEARN_EPISODES = 1000 # Set this to a large value to stop at LEARN_TIMESTEPS instead.
ENABLE_ACTION_PENALTY = True

def train(evaluate_after=True):
    env_id = "CyberBattleToyCtf-v0"
    env = gym.make( env_id,
                    defender_constraint=DefenderConstraint(maintain_sla=0.80),
                    defender_agent=PrototypeLearningDefender())

    wrapper_coordinator = WrapperCoordinator()
    attacker_wrapper = AttackerEnvWrapper(
        env,
        wrapper_coordinator=wrapper_coordinator,
        max_timesteps=ENV_MAX_TIMESTEPS,
        enable_action_penalty=ENABLE_ACTION_PENALTY)

    defender_wrapper = DefenderEnvWrapper(
        env,
        attacker_reward_store=attacker_wrapper,
        wrapper_coordinator=wrapper_coordinator,
        max_timesteps=ENV_MAX_TIMESTEPS,
        enable_action_penalty=ENABLE_ACTION_PENALTY)

    check_env(defender_wrapper)

    defender_model = PPO('MultiInputPolicy', defender_wrapper, verbose=1)
    
    attacker_agent = RandomMarlonAgent(attacker_wrapper, defender_model.num_timesteps, defender_model.n_steps)
    defender_agent = BaselineMarlonAgent(defender_model)

    learn(attacker_agent, defender_agent, total_timesteps=LEARN_TIMESTEPS, n_eval_episodes=LEARN_EPISODES)

    defender_model.save('ppo_defender.zip')

    if evaluate_after:
        evaluate(max_timesteps=ENV_MAX_TIMESTEPS)

if __name__ == '__main__':
    train(evaluate_after=True)
