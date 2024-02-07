import abc
from enum import Enum
from typing import Any, Dict
import numpy as np


class EnvironmentVariant(Enum):
    # Without neighbor info in obs
    INDEPENDENT = 1
    # With neighbor info of k neighbors in obs
    WITH_K_NEIGHBORS = 2
    # With (global) network topology and all node observations in obs
    GLOBAL = 3


def reset_and_get_sizes(env) -> int:
    """
    Resets the environment to dynamically get environment sizes:
    * number of agents
    * agent observation dimensions
    * number of nodes
    * node observation dimensions

    :return: tuple (n_agents, obs_dim, n_nodes, node_obs_dim)
    """
    agent_observation, _ = env.reset()
    node_obs = env.get_node_observation()
    return (
        agent_observation.shape[0],
        agent_observation.shape[1],
        node_obs.shape[0],
        node_obs.shape[1],
    )


class NetworkEnv(abc.ABC):
    """
    Abstract graph/network environment.
    """

    @abc.abstractmethod
    def reset(self):
        """
        Resets the environment.

        :return: tuple (agent observation, agent adjacency matrix)
        """
        ...

    @abc.abstractmethod
    def step(self, act):
        """
        Step function that advances the environment.

        :param act: action of all agents
        :returns: tuple (agent obs, agent adjacency, agent reward, agent done, info)
        """
        ...

    def get(self):
        """
        Get underlying environment without wrappers.

        :return: the environment
        """
        return self

    def get_final_info(self, info: Dict[str, Any]):
        """
        Get additional info at the end of an episode that's not included in the step
        info.

        :param info: current info dict that will be extended in-place
        :returns: updated info dict
        """
        return info

    def get_node_aux(self):
        """
        Optional auxiliary targets for each node in the network.

        :return: None (default) or auxiliary targets of shape (num_nodes, node_aux_target_size)
        """
        return None

    @abc.abstractmethod
    def get_node_agent_matrix(self) -> np.ndarray:
        """
        Get a matrix that indicates where agents are located,
        matrix[n, a] = 1 iff agent a is on node n and 0 otherwise.

        :return: the node agent matrix of shape (n_nodes, n_agents)
        """
        ...

    @abc.abstractmethod
    def get_nodes_adjacency(self) -> np.ndarray:
        """
        Get a matrix of shape (n_nodes, n_nodes) that indicates node adjacency

        :return: node adjacency matrix
        """
        ...

    @abc.abstractmethod
    def get_node_observation(self) -> np.ndarray:
        """
        Get node observations of shape (n_nodes, node_obs_dim) with dynamic but
        consistent node_obs_dim.

        :return: node observation for all nodes
        """
        ...

    @abc.abstractmethod
    def get_num_agents(self):
        """
        Get number of agents in the environment.

        :return: number of agents
        """
        ...

    @abc.abstractmethod
    def get_num_nodes(self):
        """
        Get number of nodes in the environment.

        :return: number of nodes
        """
        ...
