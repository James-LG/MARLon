from typing import Any, Dict, Tuple

import numpy as np

import gym
from cyberbattle._env.cyberbattle_env import Action, CyberBattleEnv, Observation
from cyberbattle.simulation import model
from gym import spaces

class CyberbattleEnvWrapper(gym.Env):
    nested_spaces = ['credential_cache_matrix', 'leaked_credentials']
    other_removed_spaces = ['local_vulnerability', 'remote_vulnerability', 'connect']
    int32_spaces = ['customer_data_found', 'escalation', 'lateral_move', 'newly_discovered_nodes_count', 'probe_result']

    def __init__(self, cyber_env: CyberBattleEnv):
        super().__init__()
        self.cyber_env = cyber_env
        self.bounds = self.cyber_env._bounds
        self.observation_space = self.__create_observation_space(cyber_env)
        self.action_space = self.__create_action_space(cyber_env)

        self.valid_action_count = 0
        self.invalid_action_count = 0

    def __create_observation_space(self, cyber_env: CyberBattleEnv) -> gym.Space:
        observation_space = cyber_env.observation_space.__dict__['spaces']

        # Flatten the action_mask field.
        observation_space['local_vulnerability'] = observation_space['action_mask']['local_vulnerability']
        observation_space['remote_vulnerability'] = observation_space['action_mask']['remote_vulnerability']
        observation_space['connect'] = observation_space['action_mask']['connect']
        del observation_space['action_mask']

        # Remove 'info' fields added by cyberbattle that do not represent algorithm inputs
        del observation_space['credential_cache']
        del observation_space['discovered_nodes']
        del observation_space['explored_network']

        # TODO: Reformat these spaces so they don't have to be removed
        # Remove nested Tuple/Dict spaces
        for space in self.nested_spaces + self.other_removed_spaces:
            del observation_space[space]

        # This is incorrectly set to spaces.MultiBinary(2)
        # It's a single value in the returned observations
        observation_space['customer_data_found'] = spaces.Discrete(2)

        # This is incorrectly set to spaces.MultiDiscrete(model.PrivilegeLevel.MAXIMUM + 1), when it is only one value
        observation_space['escalation'] = spaces.Discrete(model.PrivilegeLevel.MAXIMUM + 1)
        
        return spaces.Dict(observation_space)

    def __create_action_space(self, cyber_env: CyberBattleEnv) -> gym.Space:
        self.action_subspaces = {}
        # First action defines which action subspace to use
        # local_vulnerability, remote_vulnerability, or connect
        action_space = [2]

        # CyberBattle's action space is a dict of nested action spaces.
        # We need to flatten it into a single multidiscrete and keep
        # track of which values correspond to which nested values so
        # we can reconstruct the action later.
        subspace_index = 0
        for (key, value) in cyber_env.action_space.spaces.items():
            subspace_start = len(action_space)
            for vec in value.nvec:
                action_space.append(vec)

            # Action subspace takes the form:
            # [('subspace_name', 1, 3), ('subspace_name2', 3, 5)]
            self.action_subspaces[subspace_index] = (key, subspace_start, len(action_space))
            subspace_index += 1

        return spaces.MultiDiscrete(action_space)

    def __get_owned_nodes(self):
        return np.nonzero(self.cyber_env._CyberBattleEnv__get_privilegelevel_array())[0]

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        # The first action value corresponds to the subspace
        action_subspace = self.action_subspaces[action[0]]

        # Translate the flattened action back into the nested
        # subspace action for CyberBattle. It takes the form:
        # {'subspace_name': [0, 1, 2]}
        translated_action = {action_subspace[0]: action[action_subspace[1]:action_subspace[2]]}

        # For reference:
        # ```python
        # action_spaces: ActionSpaceDict = {
        #     "local_vulnerability": spaces.MultiDiscrete(
        #         # source_node_id, vulnerability_id
        #         [maximum_node_count, local_vulnerabilities_count]),
        #     "remote_vulnerability": spaces.MultiDiscrete(
        #         # source_node_id, target_node_id, vulnerability_id
        #         [maximum_node_count, maximum_node_count, remote_vulnerabilities_count]),
        #     "connect": spaces.MultiDiscrete(
        #         # source_node_id, target_node_id, target_port, credential_id
        #         # (by index of discovery: 0 for initial node, 1 for first discovered node, ...)
        #         [maximum_node_count, maximum_node_count, port_count, maximum_total_credentials])
        # }
        # ```

        # First, check if the action is valid
        if not self.cyber_env.is_action_valid(translated_action):
            # If it is not valid, we will try picking a random valid node and hoping 
            # that makes the action valid.
            
            # Pick source node at random (owned and with the desired feature encoding)
            potential_source_nodes = [
                from_node
                for from_node in self.__get_owned_nodes()
                #if np.all(actor_features == self.node_specific_features.get(wrapped_env.state, from_node))
            ]

            if len(potential_source_nodes) > 0:
                source_node = np.random.choice(potential_source_nodes)

                if action_subspace[0] == 'local_vulnerability':
                    # Replace node from the algorithm with a valid node.
                    translated_action[action_subspace[0]][0] = source_node
                else:
                    # Target node can be any potential node excluding source node.
                    potential_source_nodes.remove(source_node)

                    if len(potential_source_nodes) > 0:
                        target_node = np.random.choice(potential_source_nodes)

                        # Replace source and target node from the algorithm with valid nodes.
                        translated_action[action_subspace[0]][0] = source_node
                        translated_action[action_subspace[0]][1] = target_node
                    else:
                        # No potential target nodes
                        pass
            else:
                # No potential source nodes
                pass

        # If the action is still invalid, sample a random valid action.
        # TODO: Try invalid action masks instead of sampling a random valid action; 'Dynamic action spaces'.
        # https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
        # TODO: Give a negative reward if invalid.
        if not self.cyber_env.is_action_valid(translated_action):
            # sample local and remote actions only (excludes connect action)
            translated_action = self.cyber_env.sample_valid_action(kinds=[0, 1, 2])
            self.invalid_action_count += 1
        else:
            self.valid_action_count += 1

        observation, reward, done, info = self.cyber_env.step(translated_action)
        transformed_observation = self.transform_observation(observation)
        return transformed_observation, reward, done, info

    def reset(self) -> Observation:
        self.valid_action_count = 0
        self.invalid_action_count = 0
        observation = self.cyber_env.reset()
        return self.transform_observation(observation)

    def transform_observation(self, observation) -> Observation:
        # Flatten the action_mask field
        observation['local_vulnerability'] = observation['action_mask']['local_vulnerability']
        observation['remote_vulnerability'] = observation['action_mask']['remote_vulnerability']
        observation['connect'] = observation['action_mask']['connect']
        del observation['action_mask']

        # TODO: Retain real values
        #if observation['credential_cache_matrix'].shape == (1,2):
        credential_cache_matrix = []
        for _ in range(self.bounds.maximum_total_credentials):
            credential_cache_matrix.append(np.zeros((2,)))
        
        # TODO: Clean this up a bit, action masks are not needed here
        observation['credential_cache_matrix'] = tuple(credential_cache_matrix)
        observation['local_vulnerability'] = np.zeros((self.bounds.maximum_node_count * self.bounds.local_attacks_count,))
        observation['remote_vulnerability'] = np.zeros((self.bounds.maximum_node_count * self.bounds.maximum_node_count * self.bounds.remote_attacks_count,))
        observation['connect'] = np.zeros((self.bounds.maximum_node_count * self.bounds.maximum_node_count * self.bounds.port_count * self.bounds.maximum_total_credentials,))
        observation['discovered_nodes_properties'] = np.zeros((self.bounds.maximum_node_count * self.bounds.property_count,))
        observation['nodes_privilegelevel'] = np.zeros((self.bounds.maximum_node_count,))

        # Remove 'info' fields added by cyberbattle that do not represent algorithm inputs
        del observation['credential_cache']
        del observation['discovered_nodes']
        del observation['explored_network']

        # Stable baselines does not like numpy wrapped ints
        for space in self.int32_spaces:
            observation[space] = int(observation[space])

        # TODO: Reformat these spaces so they don't have to be removed
        # Remove nested Tuple/Dict spaces
        for space in self.nested_spaces + self.other_removed_spaces:
            del observation[space]

        return observation

    def close(self) -> None:
        return self.cyber_env.close()

    def render(self, mode: str = 'human') -> None:
        return self.cyber_env.render(mode)
