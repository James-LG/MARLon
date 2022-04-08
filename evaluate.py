
from time import sleep
from typing import Dict
import csv
from stable_baselines3 import A2C, PPO
from marlon.baseline_models.multiagent.baseline_marlon_agent import LoadFileBaselineAgentBuilder
from marlon.baseline_models.multiagent.evaluation_stats import EvalutionStats
from marlon.baseline_models.multiagent.multiagent_universe import MultiAgentUniverse
from marlon.baseline_models.multiagent.qcompat_agent import QCompatibilityAgentBuilder
from marlon.baseline_models.multiagent.random_marlon_agent import RandomAgentBuilder


def write_csv(results: Dict[str, Dict[str, EvalutionStats]]):
    print('Writing results to csv...')

    rows = []
    for attacker_name, defender_names in results.items():
        for defender_name, stats in defender_names.items():
            rows.append([
                attacker_name, defender_name,
                stats.mean_length, stats.std_length,
                stats.mean_attacker_reward, stats.std_attacker_reward,
                stats.mean_attacker_valid, stats.std_attacker_valid,
                stats.mean_attacker_invalid, stats.std_attacker_invalid,
                stats.mean_defender_reward, stats.std_defender_reward,
                stats.mean_defender_valid, stats.std_defender_valid,
                stats.mean_defender_invalid, stats.std_defender_invalid,])

    notified_wait = False
    while True:
        try:
            with open('eval_results.csv', 'w', encoding='UTF-8') as csvfile:
                csvwriter = csv.writer(csvfile)

                csvwriter.writerow(['', '', 'Episode Length', '', 'Attacker Score', '', 'Attacker Valid Actions', '', 'Attacker Invalid Actions', '', 'Defender Score', '', 'Defender Valid Actions', '', 'Defender Invalid Actions', ''])
                csvwriter.writerow(['Attacker', 'Defender', 'Mean', 'Std. Dev', 'Mean', 'Std. Dev', 'Mean', 'Std. Dev', 'Mean', 'Std. Dev', 'Mean', 'Std. Dev', 'Mean', 'Std. Dev', 'Mean', 'Std. Dev', ])

                csvwriter.writerows(rows)
                break
        except PermissionError:
            if not notified_wait:
                print('Waiting for access to file')
                notified_wait = True

            sleep(10)

    print('Done!')


def main():
    attackers = {
        'random': RandomAgentBuilder(),
        'tabularq': QCompatibilityAgentBuilder('tabularq.pkl'),
        'deepq': QCompatibilityAgentBuilder('deepq.pkl'),
        'ppo': LoadFileBaselineAgentBuilder(PPO, 'ppo.zip'),
        'ppo_marl': LoadFileBaselineAgentBuilder(PPO, 'ppo_marl_attacker.zip'),
        'a2c': LoadFileBaselineAgentBuilder(A2C, 'a2c.zip'),
        'a2c_marl': LoadFileBaselineAgentBuilder(A2C, 'a2c_marl_attacker.zip'),
    }

    defenders = {
        'none': None,
        'random': RandomAgentBuilder(),
        'ppo': LoadFileBaselineAgentBuilder(PPO, 'ppo_defender.zip'),
        'ppo_marl': LoadFileBaselineAgentBuilder(PPO, 'ppo_marl_defender.zip'),
        'a2c': LoadFileBaselineAgentBuilder(A2C, 'a2c_defender.zip'),
        'a2c_marl': LoadFileBaselineAgentBuilder(A2C, 'a2c_marl_defender.zip'),
    }

    results = {}

    for attacker_name, attacker_builder in attackers.items():
        results[attacker_name] = {}
        for defender_name, defender_builder in defenders.items():
            print('+++++++++++++++++++')
            print(f'Attacker: {attacker_name}; Defender: {defender_name}')
            universe = MultiAgentUniverse.build(
                attacker_builder=attacker_builder,
                attacker_invalid_action_reward_modifier=0,
                defender_builder=defender_builder,
                defender_invalid_action_reward_modifier=0,
            )

            results[attacker_name][defender_name] = universe.evaluate(20)

    write_csv(results)

if __name__ == "__main__":
    main()
