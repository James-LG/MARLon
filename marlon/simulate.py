from typing import List, Optional

from stable_baselines3 import PPO, A2C
import torch
from marlon.baseline_models.multiagent import marl_algorithm
from marlon.baseline_models.multiagent.baseline_marlon_agent import LoadFileBaselineAgentBuilder
from marlon.baseline_models.multiagent.multiagent_universe import MultiAgentUniverse
from marlon.baseline_models.multiagent.qcompat_agent import QCompatibilityAgentBuilder
from marlon.baseline_models.multiagent.random_marlon_agent import RandomAgentBuilder

class SimulationCache:
    value: Optional[List[str]] = None

def simulate(timesteps, attacker_option, defender_option, attacker_file, defender_file):
    if attacker_option == 'None':
        raise ValueError('Attacker cannot be none')

    attacker_builder = create_builder(attacker_option, attacker_file)
    defender_builder = create_builder(defender_option, defender_file)

    universe = MultiAgentUniverse.build(
        attacker_builder=attacker_builder,
        defender_builder=defender_builder,
        attacker_invalid_action_reward_modifier=0,
        defender_invalid_action_reward_modifier=0
    )

    _, _, simulation = marl_algorithm.run_episode(
        attacker_agent=universe.attacker_agent,
        defender_agent=universe.defender_agent,
        max_steps=timesteps,
        is_simulation=True
    )

    return simulation

def create_builder(option, file):
    if option == 'None':
        return None
    elif option == 'Random':
        builder = RandomAgentBuilder()
    elif option == 'Load':
        if 'pkl' in file:
            builder = QCompatibilityAgentBuilder(file_path=file)
        else:
            if 'ppo' in file:
                alg_type = PPO
            elif 'a2c' in file:
                alg_type = A2C
            else:
                raise ValueError(f'Baseline file {file} type could not be determined, please include `ppo` or `a2c` in the file name.')

            builder = LoadFileBaselineAgentBuilder(
                alg_type=alg_type,
                file_path=file
            )

    return builder

if __name__ == "__main__":
    simulate(2000, 'Load', 'None', 'ppo.zip', None)
