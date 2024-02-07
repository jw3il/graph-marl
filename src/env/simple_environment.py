import textwrap
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from env.environment import EnvironmentVariant, NetworkEnv
from gymnasium.spaces import Discrete


class Router(object):
    def __init__(self, x, y, s):
        self.x = x
        self.y = y
        self.neighbor = []
        self.edge = []
        self.score = s


class Edge(object):
    def __init__(self, x, y, length):
        self.start = x
        self.end = y
        self.len = int(int(length * 10) / 2 + 1)
        self.load = 0

    def get_other_node(self, node):
        if self.start == node:
            return self.end
        elif self.end == node:
            return self.start
        else:
            raise ValueError(
                f"Is neither start nor end of edge {self.start}-{self.end}: {node}"
            )


class Data(object):
    def __init__(self, x, size):
        self.now = x
        self.size = size


class SimpleEnvironment(NetworkEnv):
    """
    Simple test environment with 3 nodes connected in a line topology
    and one agent located at the nodes in the middle.

        o -- o -- o
             ^
           agent

    The agent makes a binary decision to either move to the left or right neighbor.
    These neighbor nodes have scores in {{-1, 1}} randomly chosen without replacement
    and the agent receives the score that has been assigned to the node.

    The goal is to select the action that leads to the node with score 1.
    """

    def __init__(self, env_var: EnvironmentVariant, random_topology):
        """
        Initializes the environment.

        :param env_var: Independent means that the agent only
        observes its own position, with_k_neighbors and global allow for a global view.
        :param random_topology: Whether to randomize the node ids and edge order.
        """
        self.router: List[Router] = []
        self.edges: List[Edge] = []
        self.G = nx.Graph()
        self.adj_matrix = None
        self.n_router = 3
        self.n_data = 1
        self.record_distance_map = False
        self.random_topology = random_topology
        self.sort_edges = True
        self.env_var = EnvironmentVariant(env_var)
        self.start_node = -1

        # using gym action space
        self.action_space = Discrete(2, start=0)  # {0, 1}

    @staticmethod
    def one_hot_list(i, max_indices):
        a = [0] * max_indices
        if i >= 0:
            a[i] = 1
        return a

    def get_num_agents(self):
        return self.n_data

    def get_num_nodes(self):
        return self.n_router

    def __str__(self) -> str:
        return textwrap.dedent(
            f"""\
            SimpleEnvironment with parameters
            > Environment variant: {self.env_var}
            > Random topology: {self.random_topology}\
            """
        )

    def _build_network(self):
        self.G = nx.Graph()
        self.router = []
        self.edges = []
        self.data = []

        border_scores = np.array([-1, 1])
        # border scores are always shuffled
        np.random.shuffle(border_scores)

        scores = np.array([border_scores[0], 0, border_scores[1]])
        if self.random_topology:
            # shuffle all scores
            np.random.shuffle(scores)

        # identify node ids, n0 is the start node with score 0
        n0 = np.where(scores == 0)[0][0]
        n1 = (n0 + 1) % 3
        n2 = (n1 + 1) % 3
        self.start_node = n0

        for i in range(3):
            # add routers at random locations
            new_router = Router(np.random.random(), np.random.random(), scores[i])
            self.router.append(new_router)
            self.G.add_node(i, pos=(new_router.x, new_router.y))

        self.router[n0].neighbor.append(n1)
        self.router[n0].neighbor.append(n2)
        self.router[n1].neighbor.append(n0)
        self.router[n2].neighbor.append(n0)

        edge_destinations = [n1, n2]
        if self.random_topology:
            np.random.shuffle(edge_destinations)

        edge_nodes = [n0, edge_destinations[0]]
        if self.random_topology:
            np.random.shuffle(edge_nodes)

        new_edge_0 = Edge(edge_nodes[0], edge_nodes[1], 1)
        self.edges.append(new_edge_0)

        self.G.add_edge(
            new_edge_0.start,
            new_edge_0.end,
            color="lightblue",
            weight=new_edge_0.len,
        )

        edge_nodes = [n0, edge_destinations[1]]
        if self.random_topology:
            np.random.shuffle(edge_nodes)

        new_edge_2 = Edge(edge_nodes[0], edge_nodes[1], 1)
        self.edges.append(new_edge_2)

        self.G.add_edge(
            new_edge_2.start,
            new_edge_2.end,
            color="lightblue",
            weight=new_edge_2.len,
        )

        edge_order = [0, 1]
        if self.random_topology:
            if self.sort_edges:
                edge_order = np.argsort(edge_destinations)
            else:
                np.random.shuffle(edge_order)

        self.router[n0].edge.append(edge_order[0])
        self.router[n0].edge.append(edge_order[1])

        self.router[edge_destinations[0]].edge.append(0)
        self.router[edge_destinations[1]].edge.append(1)

        # generate data packet
        self.data = []
        self.data.append(Data(self.start_node, 1))

        self._update_nodes_adjacency()

    def reset(self):
        self._build_network()
        # self.render()
        # self._network_exists = True
        return self._get_observation(), self._get_data_adjacency()

    def render(self):
        labels = {}
        for r in range(3):
            labels[r] = f"{r}:{self.router[r].score}"
        nx.draw_networkx(self.G, labels=labels, node_color="pink")
        plt.show()

    # adj matrix of routers(nodes) #
    def get_nodes_adjacency(self):
        """
        Get the adjacency matrix for all routers (nodes) in the network.

        return: adjacency matrix of size (n_router, n_router)
        """
        return self.adj_matrix

    def _update_nodes_adjacency(self):
        self.adj_matrix = np.eye(self.n_router, self.n_router, dtype=np.int8)
        for i in range(self.n_router):
            for neighbor in self.router[i].neighbor:
                self.adj_matrix[i][neighbor] = 1

    def get_node_observation(self):
        """
        Get the monitoring information for each router in the network.

        :return: monitoring info for each router
        """
        obs = []
        for j in range(self.n_router):
            ob = []
            # necessary: node score
            ob.append(self.router[j].score)

            # optional: node index
            # ob.append(j)

            # optional: edge indices
            # for edge_idx in range(2):
            #     if edge_idx >= len(self.router[j].edge):
            #         ob.append(-1)
            #     else:
            #         ob.append(self.router[j].edge[edge_idx])

            obs.append(ob)

        return np.array(obs, dtype=np.float32)

    def get_node_agent_matrix(self):
        """
        Gets a matrix that indicates where agents are located,
        matrix[n, a] = 1 iff agent a is on node n and 0 otherwise.

        :return: the node agent matrix of shape (n_nodes, n_agents)
        """
        node_agent = np.zeros((self.n_router, self.n_data), dtype=np.int8)
        for a in range(self.n_data):
            node_agent[self.data[a].now, a] = 1

        return node_agent

    def _get_observation(self):
        obs = []
        nodes_adjacency = self.get_nodes_adjacency().flatten()
        node_observation = self.get_node_observation().flatten()
        global_obs = np.concatenate((nodes_adjacency, node_observation))

        for i in range(self.n_data):
            ob = []
            # packet information
            ob.append(self.data[i].now)

            # other data
            self.data[i].neigh = []
            self.data[i].neigh.append(i)
            for j in range(self.n_data):
                if j == i:
                    continue
                if (self.data[j].now in self.router[self.data[i].now].neighbor) | (
                    self.data[j].now == self.data[i].now
                ):
                    self.data[i].neigh.append(j)

            ob_numpy = np.array(ob)

            # add global information
            if self.env_var != EnvironmentVariant.INDEPENDENT:
                # add global node observations
                ob_numpy = np.concatenate((ob_numpy, global_obs))

            obs.append(ob_numpy)

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        act = action[0]
        reward = [-1]
        done = [False]

        for i in range(self.n_data):
            # agent i controls data packet i
            packet = self.data[i]
            t = self.router[packet.now].edge[act]

            if self.edges[t].start == packet.now:
                packet.now = self.edges[t].end
            else:
                packet.now = self.edges[t].start

            reward[0] = self.router[packet.now].score
            done[0] = True

            # reset packet (middle router)
            packet.now = self.start_node

        obs = self._get_observation()
        adj = self._get_data_adjacency()
        info = {}

        # print(action, reward)
        return obs, adj, np.array(reward), done, info

    def _get_data_adjacency(self):
        """
        Get an adjacency matrix for data packets (agents) of shape (n_agents, n_agents)
        where the second dimension contains the neighbors of the agents in the first
        dimension, i.e. the matrix is of form (agent, neighbors).

        :param data: current data list
        :param n_data: number of data packets
        :return: adjacency matrix
        """
        # eye because self is also part of the neighborhood
        adj = np.eye(self.n_data, self.n_data, dtype=np.int8)
        for i in range(self.n_data):
            for n in self.data[i].neigh:
                if n != -1:
                    # n is (currently) a neighbor of i
                    adj[i, n] = 1
        return adj
