from typing import Any, Dict, Optional, Tuple, TypedDict
import boolean
import logging

import numpy as np

from plotly.missing_ipywidgets import FigureWidget

import gym
from gym import spaces
from gym.spaces.space import Space

from cyberbattle.simulation import model
from cyberbattle._env.cyberbattle_env import CyberBattleEnv, EnvironmentBounds, Observation

from marlon.baseline_models.env_wrappers.environment_event_source import IEnvironmentObserver, EnvironmentEventSource
from marlon.baseline_models.env_wrappers.reward_store import IRewardStore
from marlon.defender_agents.defender import LearningDefender


Defender_Observation = TypedDict('Defender_Observation', {'infected_nodes': np.ndarray,
                                                          'incoming_firewall_status':np.ndarray,
                                                          'outgoing_firewall_status':np.ndarray,
                                                          'services_status':np.ndarray})
class DefenderEnvWrapper(gym.Env, IEnvironmentObserver):
    """Wraps a CyberBattleEnv for stablebaselines-3 models to learn how to defend."""

    nested_spaces = ['credential_cache_matrix', 'leaked_credentials']
    other_removed_spaces = ['local_vulnerability', 'remote_vulnerability', 'connect']
    int32_spaces = ['customer_data_found', 'escalation', 'lateral_move', 'newly_discovered_nodes_count', 'probe_result']
    firewall_rule_list = ["RDP", "SSH", "HTTPS", "HTTP", "su", "sudo"]

    def __init__(self,
        cyber_env: CyberBattleEnv,
        attacker_reward_store: IRewardStore,
        event_source: Optional[EnvironmentEventSource] = None,
        defender: bool = False,
        max_timesteps=100,
        invalid_action_reward=0,
        reset_on_constraint_broken = True):

        super().__init__()
        self.defender = None
        self.cyber_env: CyberBattleEnv = cyber_env
        self.bounds: EnvironmentBounds = self.cyber_env._bounds
        self.num_services = 0
        self.observation_space: Space = self.__create_observation_space(cyber_env)
        self.action_space: Space = self.__create_defender_action_space(cyber_env)
        self.network_availability: float = 1.0
        self.valid_action_count = 0
        self.invalid_action_count = 0
        self.max_timesteps = max_timesteps
        self.timesteps = 0
        self.rewards = []
        self.attacker_reward_store = attacker_reward_store
        self.first = True
        self.reset_request = False
        self.invalid_action_penalty = invalid_action_reward
        # Add this object as an observer of the cyber env.
        if event_source is None:
            event_source = EnvironmentEventSource()

        self.event_source = event_source
        event_source.add_observer(self)
        assert defender is not None, "Attempting to use the defender environment without a defender present."
        self.defender: LearningDefender = LearningDefender(cyber_env)
        self.__last_attacker_reward = None
        self.reset_on_constraint_broken = reset_on_constraint_broken

    def __create_observation_space(self, cyber_env: CyberBattleEnv) -> gym.Space:
        """Creates a compatible version of the attackers observation space."""
        # Calculate how many services there are, this is used to define the maximum number of services active at once.
        for _, node in model.iterate_network_nodes(cyber_env.environment.network):
            for _ in node.services:
                self.num_services +=1
        # All spaces are MultiBinary.
        return spaces.Dict({'infected_nodes': spaces.MultiBinary(len(list(cyber_env.environment.network.nodes))),
                            'incoming_firewall_status': spaces.MultiBinary(len(self.firewall_rule_list)*len(list(cyber_env.environment.network.nodes))),
                            'outgoing_firewall_status': spaces.MultiBinary(len(self.firewall_rule_list)*len(list(cyber_env.environment.network.nodes))),
                            'services_status': spaces.MultiBinary(self.num_services)})

    def __create_defender_action_space(self, cyber_env: CyberBattleEnv) -> gym.Space:
        # 0th index of the action defines which action to use (reimage, block_traffic, allow_traffic, stop_service, start_service)
        # Index 1 is the possible nodes to reimage (all nodes) (Only used on action 0)
        # Index 2, 3, 4 are for action 1 (block traffic) 2nd = node to block on, 3rd =Port to block, 4th = incoming or outgoing
        # Index 5, 6, 7 relate to action 2 (allow traffic), 5th = node to allow on, 6th = Port to allow, 7th = incoming or outgoing
        # Index 8 and 9 are for action 3 (stop service), 8th = node to stop service on, 9th = port to stop service
        # Index 10 and 11 are for action 4 (start service), 10th = node to start service on, 11th = port to start service on.
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

    def step(self, action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        done = False
        reward = 0
        # Check for action validity
        if not self.is_defender_action_valid(action):
            logging.info(f"Action chosen is outside action space. Defender will skip this turn. Action = {action}")
            self.invalid_action_count += 1
            reward += self.invalid_action_penalty
            # If the action is invalid, pass an empty list to the defender
            action = []
        else:
            self.valid_action_count += 1
        
        self.defender.executeAction(action)
        # Take the reward gained this step from the attacker's step and invert it so the defender
        # loses more reward if the attacker succeeds.
        if self.attacker_reward_store.episode_rewards:
            reward += -1*self.attacker_reward_store.episode_rewards[-1]

        if self.defender_constraints_broken():
            reward = self.cyber_env._CyberBattleEnv__LOSING_REWARD
            logging.warning("Defender Lost")
            if self.reset_on_constraint_broken:
                done = True
        if self.cyber_env._CyberBattleEnv__defender_goal_reached():
            reward = self.cyber_env._CyberBattleEnv__WINNING_REWARD
            done = True
        # Generate the new defender observation based on the defender's action
        defender_observation = self.observe()
        self.timesteps += 1

        if self.reset_request:
            done = True
            reward = -1*self.__last_attacker_reward
        elif self.timesteps > self.max_timesteps:
            done = True

        self.rewards.append(reward)
        return defender_observation, reward, done, {}

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
                for rule in node_info.firewall.outgoing:
                    firewall_list.append(rule.port)

            return self.firewall_rule_list[port_from_action] in firewall_list

        def service_exists(node_info: model.NodeInfo, service_from_action: int):
            """Checks if a service exists on a node (Only checks if the service is out of bounds for the node)"""
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

        elif action_number == 2:
            # allow traffic
            _, node_info = get_node_and_info(action[5])
            # If the node does not exist, or if the node is not currently running, this action is invalid.
            return node_exists_and_running(action[5])

        elif action_number == 3:
            # stop service
            # If the node does not exist, or if the node is not currently running, this action is invalid.
            # Also if the service to stop does not exist, this is invalid
            _, node_info = get_node_and_info(action[8])
            return node_exists_and_running(action[8]) and service_exists(node_info, action[9])

        elif action_number == 4:
            # start service
            # If the node does not exist, or if the node is not currently running, this action is invalid.
            # Also if the service to start does not exist, this is invalid
            _, node_info = get_node_and_info(action[10])
            return node_exists_and_running(action[10]) and service_exists(node_info, action[11])
        else:
            return False

    def reset(self) -> Observation:
        logging.debug('Reset Defender')
        if not self.reset_request:
            self.event_source.notify_reset(last_reward=0)

        self.cyber_env.reset()

        self.reset_request = False
        self.__last_attacker_reward = None
        self.rewards = []
        self.timesteps = 0
        self.valid_action_count = 0
        self.invalid_action_count = 0

        return self.observe()

    def on_reset(self, last_reward):
        logging.debug('on_reset Defender')
        self.reset_request = True
        self.__last_attacker_reward = last_reward

    def get_blank_defender_observation(self):
        """ Creates a empty defender observation. """
        obs = Defender_Observation(infected_nodes = [],
                                    incoming_firewall_status=[],
                                    outgoing_firewall_status=[],
                                    services_status=[])
        return obs

    def observe(self) -> Defender_Observation:
        """Gathers information directly from the environment to generate populate an observation for the defender agent to use."""

        new_observation=self.get_blank_defender_observation()
        incoming_firewall_list = [0]*(len(self.cyber_env.environment.network.nodes)*len(self.firewall_rule_list))
        outgoing_firewall_list = [0]*(len(self.cyber_env.environment.network.nodes)*len(self.firewall_rule_list))
        all_services_list = [0]*self.num_services
        count_incoming_firewall = -1
        count_outgoing_firewall = -1
        count_services = -1

        # Iterates through all nodes in the environment.
        for _, node in model.iterate_network_nodes(self.cyber_env.environment.network):
            # Incoming Firewall rules section. Counts which incoming firewall rules are active.
            for rule in self.firewall_rule_list:
                count_incoming_firewall+=1
                for entry in node.firewall.incoming:
                    if rule == entry.port:
                        incoming_firewall_list[count_incoming_firewall] = 1

            # Outgoing Firewall rules section. Counts which outgoing firewall rules are active.
            for rule in self.firewall_rule_list:
                count_outgoing_firewall+=1
                for entry in node.firewall.outgoing:
                    if rule == entry.port:
                        outgoing_firewall_list[count_outgoing_firewall] = 1
                    
            # Services Section. Counts the currently running services.
            for service in node.services:
                count_services+=1
                if service.running:
                    all_services_list[count_services] = 1
                    
        # Take information from the environment and format it for defender agent observation.
        # Check all nodes and find which are infected. 1 if infected 0 if not.
        new_observation["infected_nodes"] = np.array([1 if node.agent_installed else 0 for _, node in model.iterate_network_nodes(self.cyber_env.environment.network)])
        # Lists all possible incoming firewall rules, 1 if active, 0 if not.
        new_observation['incoming_firewall_status'] = np.array(incoming_firewall_list)
        # Lists all possible outgoing firewall rules, 1 if active, 0 if not.
        new_observation['outgoing_firewall_status'] = np.array(outgoing_firewall_list)
        # Lists all possible services, 1 if active, 0 if not.
        new_observation['services_status'] = np.array(all_services_list)
        return new_observation

    def set_reset_request(self, reset_request):
        self.reset_request = reset_request

    def defender_constraints_broken(self):
        return self.cyber_env._defender_actuator.network_availability < self.cyber_env._CyberBattleEnv__defender_constraint.maintain_sla

    def close(self) -> None:
        return self.cyber_env.close()

    def render(self, mode: str = 'human') -> None:
        return self.cyber_env.render(mode)

    def render_as_fig(self) -> FigureWidget:
        return self.cyber_env.render_as_fig()
