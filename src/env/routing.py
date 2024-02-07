import textwrap
import numpy as np
from collections import defaultdict

from env.environment import EnvironmentVariant, NetworkEnv
from gymnasium.spaces import Discrete

from env.network import Network
from util import one_hot_list


class Data:
    """
    A data packet.
    """

    def __init__(self, id):
        self.id = id
        self.now = None
        self.target = None
        self.size = None
        self.start = None
        self.time = 0
        self.edge = -1
        self.neigh = None
        self.ttl = None
        self.shortest_path_weight = None
        self.visited_nodes = None

    def reset(self, start, target, size, ttl, shortest_path_weight):
        self.now = start
        self.target = target
        self.size = size
        self.start = start
        self.time = 0
        self.edge = -1
        self.neigh = [self.id]
        self.ttl = ttl
        self.shortest_path_weight = shortest_path_weight
        self.visited_nodes = set([start])


class Routing(NetworkEnv):
    """ "
    Routing environment based on the environment by
    Jiang et al. https://github.com/PKU-RL/DGN/blob/master/Routing/routers.py
    used for their DGN paper https://arxiv.org/abs/1810.09202.

    The task is to route packets from random source to random destination nodes in a
    given network. Each agent controls a single packet. When a packet reaches its
    destination, a new packet is instantly created at a random location with a new
    random target.
    """

    def __init__(
        self,
        network: Network,
        n_data,
        env_var: EnvironmentVariant,
        k=3,
        enable_congestion=True,
        enable_action_mask=False,
        ttl=0,
    ):
        """
        Initialize the environment.

        :param network: a network
        :param n_data: the number of data packets
        :param env_var: the environment variant
        :param k: include k neighbors in local observation (only for environment variant WITH_K_NEIGHBORS), defaults to 3
        :param enable_congestion: whether to respect link capacities, defaults to True
        :param enable_action_mask: whether to generate an action mask for agents that does not allow visiting nodes twice, defaults to False
        :param ttl: time to live before packets are discarded, defaults to 0
        """
        super(Routing, self).__init__()

        self.network = network
        assert isinstance(self.network, Network)

        self.n_data = n_data
        self.data = []

        # make sure env_var is casted
        self.env_var = EnvironmentVariant(env_var)

        # optionally include k neighbors in local observation
        self.k = k

        # log information
        self.agent_steps = np.zeros(self.n_data)

        # whether to use random targets or target == 0 for all packets
        self.num_random_targets = self.network.n_nodes
        assert self.num_random_targets >= 0

        # map from shortest path to actual agent steps
        self.distance_map = defaultdict(list)
        self.enable_ttl = ttl > 0
        self.enable_congestion = enable_congestion
        self.ttl = ttl
        self.sum_packets_per_node = None
        self.sum_packets_per_edge = None

        self.enable_action_mask = enable_action_mask
        self.action_mask = np.zeros((n_data, 4), dtype=bool)

        self.action_space = Discrete(4, start=0)  # {0, 1, 2, 3} using gym action space
        self.eval_info_enabled = False

    def set_eval_info(self, val):
        """
        Whether the step function should return additional info for evaluation.

        :param val: the step function returns additional info if true
        """
        self.eval_info_enabled = val

    def reset_packet(self, packet: Data):
        """
        Resets the given data packet using the settings of this environment.

        :param packet: a data packet that will be reset *in-place*
        """
        # free resources on used edge
        if packet.edge != -1:
            self.network.edges[packet.edge].load -= packet.size

        # reset packet in place
        start = np.random.randint(self.network.n_nodes)
        target = np.random.randint(self.num_random_targets)
        packet.reset(
            start=start,
            target=target,
            size=np.random.random(),
            ttl=self.ttl,
            shortest_path_weight=self.network.shortest_paths_weights[start][target],
        )

        if self.enable_action_mask:
            # all links are allowed
            self.action_mask[packet.id] = 0
            # idling is allowed if a packet spawns at the destination
            self.action_mask[packet.id, 0] = packet.now != packet.target

    def __str__(self) -> str:
        return textwrap.dedent(
            f"""\
            Routing environment with parameters
            > Network: {self.network.n_nodes} nodes
            > Number of packets: {self.n_data}
            > Environment variant: {self.env_var.name}
            > Number of considered neighbors (k): {self.k if self.env_var == EnvironmentVariant.WITH_K_NEIGHBORS else "disabled"}
            > Congestion: {self.enable_congestion}
            > Action mask: {self.enable_action_mask}
            > TTL: {self.ttl if self.enable_ttl else "disabled"}\
            """
        )

    def reset(self):
        self.agent_steps = np.zeros(self.n_data)
        self.network.reset()
        for edge in self.network.edges:
            # add new load attribute to edges
            edge.load = 0

        if self.eval_info_enabled:
            self.sum_packets_per_node = np.zeros(self.network.n_nodes)
            self.sum_packets_per_edge = np.zeros(len(self.network.edges))

        # generate random data packets
        self.data = []
        for i in range(self.n_data):
            new_data = Data(i)
            self.reset_packet(new_data)
            self.data.append(new_data)

        return self._get_observation(), self._get_data_adjacency()

    def render(self):
        # TODO: also render packets
        self.network.render()

    def get_nodes_adjacency(self):
        return self.network.adj_matrix

    def get_node_observation(self):
        """
        Get the node observation for each node in the network.

        :return: node observations of shape (num_nodes, node_observation_size)
        """
        obs = []
        for j in range(self.network.n_nodes):
            ob = []

            # router info
            # ob.append(j)
            ob += one_hot_list(j, self.network.n_nodes)
            num_packets = 0
            total_load = 0
            for i in range(self.n_data):
                if self.data[i].now == j and self.data[i].edge == -1:
                    num_packets += 1
                    total_load += self.data[i].size

            # for dest in range(self.n_router):
            #     ob.append(self.shortest_paths_weights[j][dest])

            ob.append(num_packets)
            ob.append(total_load)

            # #position obs
            # ob.append(self.router[j].y)
            # ob.append(self.router[j].x)

            # my_path_to_zero = self.shortest_paths[j][0]
            # next_node = my_path_to_zero[1] if len(my_path_to_zero) > 1 else -1

            # edge info
            for k in self.network.nodes[j].edges:
                other_node = self.network.edges[k].get_other_node(j)
                # ob.append(other_node)
                ob += one_hot_list(other_node, self.network.n_nodes)
                ob.append(self.network.edges[k].length)
                ob.append(self.network.edges[k].load)

                # cheating: add observation that tells the node how to get to 0
                # if self.edges[k].get_other_node(j) == next_node:
                #     ob.append(1)
                # else:
                #     ob.append(0)

            obs.append(ob)
        return np.array(obs, dtype=np.float32)

    def get_node_aux(self):
        """
        Auxiliary targets for each node in the network.

        :return: Auxiliary targets of shape (num_nodes, node_aux_target_size)
        """
        aux = []
        for j in range(self.network.n_nodes):
            aux_j = []

            # for routing, it is essential for a node to estimate the distance to
            # other nodes -> auxiliary target is length of shortest paths to all nodes
            for k in range(self.network.n_nodes):
                aux_j.append(self.network.shortest_paths_weights[j][k])

            aux.append(aux_j)

        return np.array(aux, dtype=np.float32)

    def get_node_agent_matrix(self):
        """
        Gets a matrix that indicates where agents are located,
        matrix[n, a] = 1 iff agent a is on node n and 0 otherwise.

        :return: the node agent matrix of shape (n_nodes, n_agents)
        """
        node_agent = np.zeros((self.network.n_nodes, self.n_data), dtype=np.int8)
        for a in range(self.n_data):
            node_agent[self.data[a].now, a] = 1

        return node_agent

    def _get_observation(self):
        obs = []
        if self.env_var == EnvironmentVariant.GLOBAL:
            # for the global observation
            nodes_adjacency = self.get_nodes_adjacency().flatten()
            node_observation = self.get_node_observation().flatten()
            global_obs = np.concatenate((nodes_adjacency, node_observation))

        for i in range(self.n_data):
            ob = []
            # packet information
            # ob.append(self.data[i].now)
            ob += one_hot_list(self.data[i].now, self.network.n_nodes)
            # ob.append(self.data[i].target)
            ob += one_hot_list(self.data[i].target, self.network.n_nodes)

            # packets should know where they are coming from when traveling on an edge
            ob.append(int(self.data[i].edge != -1))
            if self.data[i].edge != -1:
                other_node = self.network.edges[self.data[i].edge].get_other_node(
                    self.data[i].now
                )
            else:
                other_node = -1
            ob += one_hot_list(other_node, self.network.n_nodes)

            ob.append(self.data[i].time)
            ob.append(self.data[i].size)
            ob.append(self.data[i].id)

            # edge information
            for j in self.network.nodes[self.data[i].now].edges:
                other_node = self.network.edges[j].get_other_node(self.data[i].now)
                # ob.append(other_node)
                ob += one_hot_list(other_node, self.network.n_nodes)
                ob.append(self.network.edges[j].length)
                ob.append(self.network.edges[j].load)

                # ob.append(self.shortest_paths_weights[other_node][self.data[i].target])
                # for dest in range(self.n_router):
                #     ob.append(dest == self.data[i].target)
                #     ob.append(
                #         1
                #         * (dest == self.data[i].target)
                #         * self.shortest_paths_weights[other_node][dest]
                #     )

            # other data
            count = 0
            self.data[i].neigh = []
            self.data[i].neigh.append(i)
            for j in range(self.n_data):
                if j == i:
                    continue
                if (
                    self.data[j].now in self.network.nodes[self.data[i].now].neighbors
                ) | (self.data[j].now == self.data[i].now):
                    self.data[i].neigh.append(j)

                    # with neighbor information in observation (until k neighbors)
                    if (
                        self.env_var == EnvironmentVariant.WITH_K_NEIGHBORS
                        and count < self.k
                    ):
                        count += 1
                        ob.append(self.data[j].now)
                        ob.append(self.data[j].target)
                        ob.append(self.data[j].edge)
                        ob.append(self.data[j].size)
                        ob.append(self.data[i].id)

            if self.env_var == EnvironmentVariant.WITH_K_NEIGHBORS:
                for j in range(self.k - count):
                    for _ in range(5):
                        ob.append(-1)  # invalid placeholder

            # for j in range(self.n_router):
            #     # cooridnates info
            #     ob.append(self.router[j].y)
            #     ob.append(self.router[j].x)

            ob_numpy = np.array(ob)

            # add global information
            if self.env_var == EnvironmentVariant.GLOBAL:
                ob_numpy = np.concatenate((ob_numpy, global_obs))

            obs.append(ob_numpy)

        return np.array(obs, dtype=np.float32)

    def step(self, act):
        reward = np.zeros(self.n_data, dtype=np.float32)
        looped = np.zeros(self.n_data, dtype=np.float32)
        done = np.zeros(self.n_data, dtype=bool)
        drop_packet = np.zeros(self.n_data, dtype=bool)
        success = np.zeros(self.n_data, dtype=bool)
        blocked = 0

        delays = []
        delays_arrived = []
        spr = []
        self.agent_steps += 1

        # optionally shuffle packet order so that lower packet ids
        # are not prioritized anymore
        # random_packet_order = np.arange(self.n_data)
        # np.random.shuffle(random_packet_order)

        # handle actions
        # for i in random_packet_order:
        for i in range(self.n_data):
            # agent i controls data packet i
            packet = self.data[i]

            if self.eval_info_enabled:
                if packet.edge == -1:
                    self.sum_packets_per_node[packet.now] += 1

            # select outgoing edge (act == 0 is idle)
            if packet.edge == -1 and act[i] != 0:
                t = self.network.nodes[packet.now].edges[act[i] - 1]
                # note that packets that are handled earlier in this loop
                # (i.e. with lower ids) are prioritized here.
                if (
                    self.enable_congestion
                    and self.network.edges[t].load + packet.size > 1
                ):
                    # not possible to take this edge => collision
                    reward[i] -= 0.2
                    blocked += 1
                else:
                    # take this edge
                    packet.edge = t
                    packet.time = self.network.edges[t].length
                    # assign load to the selected edge
                    self.network.edges[t].load += packet.size

                    # already set the next position
                    packet.now = self.network.edges[t].get_other_node(packet.now)
                    if packet.now in packet.visited_nodes:
                        looped[i] = 1
                    else:
                        packet.visited_nodes.add(packet.now)

        if self.eval_info_enabled:
            total_edge_load = 0
            occupied_edges = 0
            packets_on_edges = 0
            total_packet_size = 0
            packet_sizes = []

            for edge in self.network.edges:
                total_edge_load += edge.load
                if edge.load > 0:
                    occupied_edges += 1

            for i in range(self.n_data):
                packet = self.data[i]
                if packet.edge != -1:
                    self.sum_packets_per_edge[packet.edge] += 1

                total_packet_size += packet.size
                packet_sizes.append(self.data[i].size)
                if packet.edge != -1:
                    packets_on_edges += 1

            packet_distances = list(
                map(
                    lambda p: self.network.shortest_paths_weights[p.now][p.target],
                    self.data,
                )
            )

        # then simulate in-flight packets (=> effect of actions)
        for i in range(self.n_data):
            packet = self.data[i]
            packet.ttl -= 1

            if packet.edge != -1:
                packet.time -= 1
                # the packet arrived at the destination, reduce load from edge
                if packet.time <= 0:
                    self.network.edges[packet.edge].load -= packet.size
                    packet.edge = -1

            drop_packet[i] = drop_packet[i] or (self.enable_ttl and packet.ttl <= 0)
            if self.enable_action_mask:
                if packet.edge != -1:
                    self.action_mask[i] = 0
                else:
                    self.action_mask[i, 0] = 1
                    for edge_i, e in enumerate(self.network.nodes[packet.now].edges):
                        self.action_mask[i, 1 + edge_i] = (
                            self.network.edges[e].get_other_node(packet.now)
                            in packet.visited_nodes
                        )

                    # packets that can't do anything are dropped
                    if self.action_mask[i].sum() == 4:
                        drop_packet[i] = True

            # the packet has reached the target
            has_reached_target = packet.edge == -1 and packet.now == packet.target
            if has_reached_target or drop_packet[i]:
                reward[i] += 10 if has_reached_target else -10
                done[i] = True
                success[i] = has_reached_target

                # we need at least 1 step (idle) if we spawn at the target
                opt_distance = max(packet.shortest_path_weight, 1)

                # insert delays before resetting packets
                if success[i]:
                    delays_arrived.append(self.agent_steps[i])
                    spr.append(self.agent_steps[i] / opt_distance)
                    if self.eval_info_enabled:
                        self.distance_map[opt_distance].append(self.agent_steps[i])

                delays.append(self.agent_steps[i])

                self.agent_steps[i] = 0
                self.reset_packet(packet)
            # else:
            #     # negative reward for distance in hops
            #     distance = len(self.shortest_paths[packet.now][packet.target])
            #     reward[i] -= distance * 0.01

        obs = self._get_observation()
        adj = self._get_data_adjacency()
        info = {
            "delays": delays,
            "delays_arrived": delays_arrived,
            # shortest path ratio in [1, inf) where 1 is optimal
            "spr": spr,
            "looped": looped.sum(),
            "throughput": success.sum(),
            "dropped": (done & ~success).sum(),
            "blocked": blocked,
        }
        if self.eval_info_enabled:
            info.update(
                {
                    "total_edge_load": total_edge_load,
                    "occupied_edges": occupied_edges,
                    "packets_on_edges": packets_on_edges,
                    "total_packet_size": total_packet_size,
                    "packet_sizes": packet_sizes,
                    "packet_distances": packet_distances,
                }
            )
        return obs, adj, reward, done, info

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

    def get_final_info(self, info: dict):
        agent_steps = self.agent_steps
        for agent_step in agent_steps:
            if agent_step != 0:
                info["delays"].append(agent_step)
        return info

    def get_num_agents(self):
        return self.n_data

    def get_num_nodes(self):
        return self.network.n_nodes
