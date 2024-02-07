from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.classes.function import path_weight
from collections import defaultdict


class Node:
    """
    A node in a network.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.neighbors = []
        self.edges = []


class Edge:
    """
    An edge in a network.
    """

    def __init__(self, start, end, length):
        self.start = start
        self.end = end
        self.length = length

    def get_other_node(self, node):
        if self.start == node:
            return self.end
        elif self.end == node:
            return self.start
        else:
            raise ValueError(
                f"Is neither start nor end of edge {self.start}-{self.end}: {node}"
            )


class Network:
    """
    Network class that manages the creation of graphs.
    """

    def __init__(
        self,
        n_nodes=20,
        random_topology=False,
        n_random_seeds=None,
        sequential_topology_seeds=False,
        topology_init_seed=476,
        excluded_seeds: Optional[List[int]] = None,
        provided_seeds: Optional[List[int]] = None,
    ):
        """
        Initializes the network and optionally creates list of valid random topology seeds.

        :param n_nodes: Number of nodes in the network, defaults to 20
        :param random_topology: Create a random topology on each reset, defaults to False
        :param n_random_seeds: Number of random topologies, defaults to None
        :param sequential_topology_seeds: Sample topologies sequentially, defaults to False
        :param topology_init_seed: Seed for topology creation, defaults to 476
        :param excluded_seeds: Seeds that must not be used for topology creation, defaults to None
        :param provided_seeds: Use provided seeds and generate no new topologies, defaults to None
        """
        self.n_nodes = n_nodes

        self.nodes = []
        self.edges = []

        self.G = nx.Graph()
        self.G_weight_key = "weight"  # None for hops or "weight" for lengths
        self.shortest_paths = None
        self.shortest_paths_weights = None
        self.adj_matrix = None

        # only needed for evaluate the efficiency of creating a correct random topology
        self.repetitions = 0

        self.random_topology = random_topology
        self.current_topology_seed = None
        self.sampled_topology_seeds = []
        self.sequential_topology_seeds = sequential_topology_seeds
        self.sequential_topology_seeds_frozen = False
        self.sequential_topology_index = 0
        self.topology_init_seed = topology_init_seed
        self.exclude_seeds = None if excluded_seeds is None else set(excluded_seeds)
        self.provide_seeds = provided_seeds
        if provided_seeds is not None and len(provided_seeds) > 0:
            self.seeds = provided_seeds
        else:
            self.seeds = self.build_seed_list(
                random_topology, n_random_seeds, self.exclude_seeds
            )
        if self.exclude_seeds is not None:
            assert all([s not in self.exclude_seeds for s in self.seeds])

    def build_seed_list(self, random_topology, n_random_seeds, exclude_seeds=None):
        if not random_topology:
            return [self.topology_init_seed]

        if n_random_seeds is None or n_random_seeds <= 0:
            return []

        old_rand_state = np.random.get_state()
        np.random.seed(self.topology_init_seed)

        seed_list = []
        # build list of unique num_random_seeds > 0 seeds
        while len(seed_list) < n_random_seeds:
            # build one random network and add topology seed
            new_seed = self._create_valid_network(seeds_exclude=exclude_seeds)
            if new_seed not in seed_list:
                seed_list.append(new_seed)

        np.random.set_state(old_rand_state)

        return seed_list

    def _create_random_topology(self):
        """
        Creates a new random topology. This is based on the implementation
        by Jiang et al. https://github.com/PKU-RL/DGN/blob/master/Routing/routers.py
        used for their DGN paper https://arxiv.org/abs/1810.09202.
        """
        # otherwise create a new topology
        self.G = nx.Graph()
        self.nodes = []
        self.edges = []
        t_edge = 0

        for i in range(self.n_nodes):
            # add routers at random locations
            new_router = Node(np.random.random(), np.random.random())
            self.nodes.append(new_router)
            self.G.add_node(i, pos=(new_router.x, new_router.y))

        for i in range(self.n_nodes):
            # calculate (squared) distances to all other routers
            self.dis = []
            for j in range(self.n_nodes):
                self.dis.append(
                    [
                        (self.nodes[j].x - self.nodes[i].x) ** 2
                        + (self.nodes[j].y - self.nodes[i].y) ** 2,
                        j,
                    ]
                )

            # sort by distance
            self.dis.sort(key=lambda x: x[0], reverse=False)

            # find new neighbors
            # exclude index 0 as we always have distance 0 to ourselves
            for j in range(1, self.n_nodes):
                # we have found enough neighbors => break
                if len(self.nodes[i].neighbors) == 3:
                    break

                # check for neighbor candidates
                candidate_sq_dist, candidate_idx = self.dis[j]
                if (
                    len(self.nodes[candidate_idx].neighbors) < 3
                    and i not in self.nodes[candidate_idx].neighbors
                ):
                    # append new neighbor
                    self.nodes[i].neighbors.append(candidate_idx)
                    self.nodes[candidate_idx].neighbors.append(i)

                    # create edges, always sorted by index
                    edge_distance = int(int(np.sqrt(candidate_sq_dist) * 10) / 2 + 1)
                    if i < candidate_idx:
                        new_edge = Edge(i, candidate_idx, edge_distance)
                    else:
                        new_edge = Edge(candidate_idx, i, edge_distance)

                    self.edges.append(new_edge)
                    self.nodes[candidate_idx].edges.append(t_edge)
                    self.nodes[i].edges.append(t_edge)
                    self.G.add_edge(
                        new_edge.start,
                        new_edge.end,
                        weight=new_edge.length,
                    )

                    t_edge += 1

        # order router edges by neighbor node id to remove symmetries
        for i in range(self.n_nodes):
            self.nodes[i].edges = sorted(
                self.nodes[i].edges,
                key=lambda edge_index: self.edges[edge_index].get_other_node(i),
            )

    def _check_topology_constraints(self):
        """
        Check if the current network topology fulfills the constraints, meaning it is
        connected and all nodes have three neighbors.

        :return: whether the topology is valid.
        """
        # for the case that there is no isolated island but nodes with less than k edges
        for i in range(self.n_nodes):
            if len(self.nodes[i].neighbors) < 3:
                return False

        # this means not every nodes is reachable, we have got isolated islands
        if not nx.is_connected(self.G):
            return False

        return True

    def _create_valid_network(
        self, seed_list=None, seed_index=None, seeds_exclude=None
    ):
        """
        Generates a network based on a list of seeds.

        :param seed_list: List of seeds, can be None to create new valid topology
        :param seed_index: Index in seed list, chooses random index if None
        :param seeds_exclude: List of seeds (for random generation) are excluded
        :returns: the seed used to generate the network topology
        """

        # set seed for topology generation
        no_seed_provided = seed_list is None or len(seed_list) == 0
        if no_seed_provided:
            topology_seed = np.random.randint(2**31 - 1)
            while seeds_exclude is not None and topology_seed in seeds_exclude:
                topology_seed = np.random.randint(2**31 - 1)

        elif seed_index is not None:
            topology_seed = seed_list[seed_index]
        else:
            # choose one of the seeds from the list
            topology_seed = np.random.choice(seed_list)

        # remember random state for packet generation
        old_rand_state = np.random.get_state()
        np.random.seed(topology_seed)

        self.repetitions = 0
        while True:
            self._create_random_topology()
            self.repetitions += 1
            if self._check_topology_constraints():
                break

            assert no_seed_provided, f"Provided seed {topology_seed} is invalid."
            topology_seed = np.random.randint(2**31 - 1)
            while seeds_exclude is not None and topology_seed in seeds_exclude:
                topology_seed = np.random.randint(2**31 - 1)
            np.random.seed(topology_seed)

        # restore old random state
        np.random.set_state(old_rand_state)

        # calculate shortest paths with corresponding distances/weights
        self._update_shortest_paths()

        # e_lens = np.array([e.len for e in self.edges])
        # print(
        #     f"Max: {e_lens.max()}, min: {e_lens.min()}, mean {e_lens.mean()}, std {e_lens.std()}"
        # )

        self._update_nodes_adjacency()
        self.current_topology_seed = topology_seed

        # return the seed that was used to create this topology
        return topology_seed

    def _update_shortest_paths(self):
        """
        Calculates shortest paths and stores them in self.shortest_paths. The
        corresponding weights (distances) are stored in self.shortest_paths_weights
        """
        self.shortest_paths = dict(nx.shortest_path(self.G, weight=self.G_weight_key))
        self.shortest_paths_weights = defaultdict(dict)
        for start in self.shortest_paths:
            for end in self.shortest_paths[start]:
                if self.G_weight_key is None:
                    self.shortest_paths_weights[start][end] = (
                        len(self.shortest_paths[start][end]) - 1
                    )
                else:
                    self.shortest_paths_weights[start][end] = path_weight(
                        self.G, self.shortest_paths[start][end], self.G_weight_key
                    )

    def randomize_edge_weights(self, mode: str, **kwargs):
        """
        Randomizes edge weights in the graph (at runtime).

        :param mode: `shuffle` to shuffle existing weights, `randint` with additional
                     kwargs `low` and `high` to create new random weights
        :returns: tuple of (proportion of changed first hops on shortest paths, proportion
                  of changed shortest paths, proportion of changed shortest path lengths)
        """
        if mode == "shuffle":
            edge_lengths = np.array([e.length for e in self.edges])
            np.random.shuffle(edge_lengths)
            for i, e in enumerate(self.edges):
                e.length = edge_lengths[i]
        elif mode == "randint":
            for e in self.edges:
                e.length = np.random.randint(kwargs["low"], kwargs["high"])
        elif mode == "bottleneck-971182936":
            edge_update_list = [
                (2, 7),
            ]
            for e in self.edges:
                for (start, end) in edge_update_list:
                    if e.start == start and e.end == end:
                        e.length = 10
                        break
            if self.current_topology_seed != 971182936:
                print("Warning: mode only meant to be used in graph 971182936.")
        else:
            raise ValueError(f"Unknown mode {mode}")

        old_shortest_paths = self.shortest_paths.copy()
        old_shortest_path_weights = self.shortest_paths_weights.copy()

        for e in self.edges:
            self.G[e.start][e.end]["weight"] = e.length

        self._update_shortest_paths()

        # check how much has changed
        n_paths = self.n_nodes * (self.n_nodes - 1)
        n_paths_changed_first_hop = 0
        n_paths_changed = 0
        n_path_weights_changed = 0
        for a in range(self.n_nodes):
            for b in range(self.n_nodes):
                if a == b:
                    continue
                if self.shortest_paths[a][b][1] != old_shortest_paths[a][b][1]:
                    n_paths_changed_first_hop += 1
                if self.shortest_paths[a][b] != old_shortest_paths[a][b]:
                    n_paths_changed += 1
                if self.shortest_paths_weights[a][b] != old_shortest_path_weights[a][b]:
                    n_path_weights_changed += 1

        return (
            n_paths_changed_first_hop / n_paths,
            n_paths_changed / n_paths,
            n_path_weights_changed / n_paths,
        )

    def freeze_sequential_topology_seeds(self):
        self.sequential_topology_seeds_frozen = True

    def next_topology_seed_index(self, advance_index=True):
        seed_index = (
            self.sequential_topology_index
            if len(self.seeds) > 1 and self.sequential_topology_seeds
            else None
        )
        if seed_index is not None and advance_index:
            self.sequential_topology_index = (seed_index + 1) % len(self.seeds)
        return seed_index

    def reset(self):
        seed_index = self.next_topology_seed_index(
            advance_index=not self.sequential_topology_seeds_frozen
        )
        self._create_valid_network(self.seeds, seed_index, self.exclude_seeds)
        self.sampled_topology_seeds.append(self.current_topology_seed)

    def render(self):
        nx.draw_networkx(self.G, with_labels=True, node_color="pink")
        plt.show()

    def get_nodes_adjacency(self):
        """
        Get the adjacency matrix for all routers (nodes) in the network.

        return: adjacency matrix of size (n_router, n_router)
        """
        return self.adj_matrix

    def _update_nodes_adjacency(self):
        self.adj_matrix = np.eye(self.n_nodes, self.n_nodes, dtype=np.int8)
        for i in range(self.n_nodes):
            for neighbor in self.nodes[i].neighbors:
                self.adj_matrix[i][neighbor] = 1
