from typing import Any, Dict, Tuple
import boolean
import cyberbattle
from gym.spaces.space import Space

import numpy as np

import gym
from cyberbattle._env.cyberbattle_env import CyberBattleEnv, EnvironmentBounds, Observation
from cyberbattle.simulation import model
from gym import spaces
from plotly.missing_ipywidgets import FigureWidget
import logging

class DefenderEnvWrapper(gym.Env):
    '''
    Wraps a CyberBattleEnv for stablebaselines-3 models to learn how to defend.
    '''

    nested_spaces = ['credential_cache_matrix', 'leaked_credentials']
    other_removed_spaces = ['local_vulnerability', 'remote_vulnerability', 'connect']
    int32_spaces = ['customer_data_found', 'escalation', 'lateral_move', 'newly_discovered_nodes_count', 'probe_result']
    firewall_rule_list = ["RDP", "SSH", "HTTPS", "HTTP", "su", "sudo"]

    def __init__(self, cyber_env: CyberBattleEnv, max_timesteps=100, enable_action_penalty=True):
        super().__init__()
        self.cyber_env: CyberBattleEnv = cyber_env
        self.bounds: EnvironmentBounds = self.cyber_env._bounds
        self.observation_space: Space = self.__create_observation_space(cyber_env)
        self.action_space: Space = self.__create_defender_action_space(cyber_env)
        self.__get_privilegelevel_array = cyber_env._CyberBattleEnv__get_privilegelevel_array
        self.valid_action_count = 0
        self.invalid_action_count = 0
        self.max_timesteps = max_timesteps
        self.timesteps = 0
        self.rewards = []

    def __create_observation_space(self, cyber_env: CyberBattleEnv) -> gym.Space:
        """Creates a compatible version of the attackers observation space."""
        # TODO Change to defender view.
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

    def __create_defender_action_space(self, cyber_env: CyberBattleEnv) -> gym.Space:
        # 0th index of the action defines which action to use (reimage, block_traffic, allow_traffic, stop_service, start_service)
        # Index 1 is the possible nodes to reimage (all nodes) (Only used on action 0)
        # Index 2, 3, 4 are for action 1 (block traffic) 2nd = node to block on, 3rd =Port to block, 4th = incoming or outgoing
        # Index 5, 6, 7 relate to action 2 (allow traffic), 5th = node to allow on, 6th = Port to allow, 7th = incoming or outgoing
        # Index 8 and 9 are for action 3 (stop service), 8th = node to stop service on, 9th = port to stop service
        # Index 10 and 11 are for action 4 (start service), 10th = node to start service on, 11th = port to start service on.
        # TODO Clean all this up
        total_actions = 5
        reimage_node_number = len(cyber_env.environment.network.nodes)
        block_traffic_node = len(cyber_env.environment.network.nodes)
        block_traffic_port = 6
        block_traffic_incoming = 2
        allow_traffic_node = len(cyber_env.environment.network.nodes)
        allow_traffic_port = 6
        allow_traffic_incoming = 2
        stop_service_node = len(cyber_env.environment.network.nodes)
        stop_service_port = 3
        start_service_node = len(cyber_env.environment.network.nodes)
        start_service_port = 3
        action_space = [total_actions, reimage_node_number, block_traffic_node, block_traffic_port, block_traffic_incoming, allow_traffic_node, allow_traffic_port, allow_traffic_incoming, stop_service_node, stop_service_port, start_service_node, start_service_port]
        logging.info(f"Action space defender = {action_space}")
        return spaces.MultiDiscrete(action_space)

    def __get_owned_nodes(self):
        return np.nonzero(self.__get_privilegelevel_array())[0]

    def step(self, action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        # Check for action validity
        if not self.is_defender_action_valid(action):
            logging.warning(f"Action choosen is outside action space. Defender will skip this turn. Action = {action}")
            self.invalid_action_count += 1
            # If the action is invalid, pass an empty list to the defender
            action = []
        else:
            self.valid_action_count += 1
        # Tell the defender which action was choosen
        self.cyber_env._CyberBattleEnv__defender_agent.next_action = action

        # Currently the defender is playing against a random agent.
        attacker_action = self.cyber_env.sample_valid_action(kinds=[0, 1, 2])
        
        # Execute the step
        observation, reward, done, info = self.cyber_env.step(attacker_action)
        transformed_observation = self.transform_observation(observation)
        self.timesteps += 1
        if self.timesteps > self.max_timesteps:
            done = True
        self.rewards.append(reward)
        return transformed_observation, reward, done, info

    def is_defender_action_valid(self, action) -> boolean:
        """Determines if a given action is valid within the environment."""
        
        def get_node_and_info(node_from_action: int):
            """Returns the node id and info for a given node"""
            node_id = get_node_from_action(node_from_action)
            node_info = get_node_info(node_id)
            return node_id, node_info

        def get_node_from_action(node_from_action: int):
            """Gets the node id from an action"""
            return list(self.cyber_env.environment.network.nodes)[node_from_action]

        def get_node_info(node_id: model.NodeID):
            """Given a node ID, find the corresponding node info"""
            return self.cyber_env.environment.get_node(node_id)


        def node_exists(node_id: model.NodeID):
            """Determines if a node exists in the network"""
            return node_id in list(self.cyber_env.environment.network.nodes)

        def node_running(node_info: model.NodeInfo):
            """Determines if a node is currently running"""
            return node_info.status == model.MachineStatus.Running

        def node_exists_and_running(node_from_action: int):
            """Determines if a node exists in the network, and if so if it is running."""
            node_id, node_info = get_node_and_info(node_from_action)
            return (node_exists(node_id) and node_running(node_info))

        def is_reimagable(node_info: model.NodeInfo):
            """Checks if a given node is reimagable"""
            return node_info.reimagable
        def firewall_rule_exists(node_info: model.NodeInfo, port_from_action: int, incoming :bool):
            """Checks a node to see if a given firewall rule exists on it."""
            firewall_list = []
            if incoming:
                for rule in node_info.firewall.incoming:
                    firewall_list.append(rule.port)
            else:
                for rule in node_info.firewall.incoming:
                    firewall_list.append(rule.port)

            return self.firewall_rule_list[port_from_action] in firewall_list

        def service_exists(node_info: model.NodeInfo, service_from_action: int):
            """Checks if a service exists on a node"""
            # TODO IS THIS GOOD? This is just checking if its out of bounds...
            # The names could be in a different order, so action 1 could be one service on one node and a differnet on another...
            return service_from_action < len(node_info.services)
        action_number = action[0]
        if action_number == 0:
            # REIMAGE
            _, node_info = get_node_and_info(action[1])
            # If the node does not exist, or if the node is not currently running, this action is invalid.
            return node_exists_and_running(action[1]) and is_reimagable(node_info)

        elif action_number == 1:
            # block traffic
            _, node_info = get_node_and_info(action[2])
            # If the node does not exist, or if the node is not currently running, this action is invalid.
            # The firewall rule needs to exist as well to block the traffic.
            return node_exists_and_running(action[2]) and firewall_rule_exists(node_info, action[3], bool(action[4]))

        elif action[0] == 2:
            # allow traffic
            _, node_info = get_node_and_info(action[5])
            # If the node does not exist, or if the node is not currently running, this action is invalid.
            return node_exists_and_running(action[5])

        elif action[0] == 3:
            # stop service
            # If the node does not exist, or if the node is not currently running, this action is invalid.
            # Also if the service to stop does not exist, this is invalid
            _, node_info = get_node_and_info(action[8])
            return node_exists_and_running(action[8]) and service_exists(node_info, action[9])

        elif action[0] == 4:
            # start service
            # If the node does not exist, or if the node is not currently running, this action is invalid.
            # Also if the service to start does not exist, this is invalid
            _, node_info = get_node_and_info(action[10])
            return node_exists_and_running(action[10]) and service_exists(node_info, action[11])
        else:
            return False

    def reset(self) -> Observation:
        self.valid_action_count = 0
        self.invalid_action_count = 0
        observation = self.cyber_env.reset()
        self.rewards = []
        self.timesteps = 0
        return self.transform_observation(observation)

    def transform_observation(self, observation) -> Observation:
        # TODO Change to defender view.
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

    def render_as_fig(self) -> FigureWidget:
        return self.cyber_env.render_as_fig()
