import numpy as np
import torch
from gymnasium.spaces import Discrete
from env.routing import Routing


class EpsilonGreedy:
    def __init__(self, env, model, action_space, args) -> None:
        self._env = env
        self._enable_action_mask = (
            hasattr(self._env, "enable_action_mask") and self._env.enable_action_mask
        )
        self._model = model
        self._action_space = action_space
        self._args = args
        self._epsilon = args.epsilon
        self._step = 0
        self._epsilon_tmp = None

    def __call__(self, obs, adj):
        self._step += 1
        # first dimension is number of agents
        actions = np.zeros(obs.shape[0], dtype=np.int32)

        with torch.no_grad():
            device = next(self._model.parameters()).device
            obs = (
                torch.tensor(obs, dtype=torch.float32)
                .unsqueeze(0)
                .to(device, non_blocking=True)
            )
            adj = (
                torch.tensor(adj, dtype=torch.float32)
                .unsqueeze(0)
                .to(device, non_blocking=True)
            )
            # run our model
            q_values = self._model(obs, adj)
            # squeeze batch dimension, our batch size is 1
            q_values = q_values.cpu().squeeze(0).detach().numpy()

            if self._enable_action_mask:
                q_values[self._env.action_mask.nonzero()] = float("-inf")

        # epsilon-greedy action selection
        random_actions = np.random.randint(self._action_space, size=actions.shape[0])
        random_filter = np.random.rand(actions.shape[0]) < self._epsilon
        actions = (
            np.argmax(q_values, axis=-1) * ~random_filter
            + random_filter * random_actions
        )

        # instead of having step as parameter we should log the steps to track
        # decay from https://github.com/PKU-RL/DGN/blob/master/Routing/routers_regularization.py
        if (
            self._epsilon > 0
            and self._step > self._args.step_before_train
            and self._step % self._args.epsilon_update_freq == 0
        ):
            self._epsilon *= self._args.epsilon_decay
            if self._epsilon < 0.01:
                self._epsilon = 0.01

        return actions

    def eval(self):
        # remember epsilon and switch to greedy policy
        self._eps_tmp = self._epsilon
        self._epsilon = 0

    def reset(self, agents_to_reset):
        """
        Resets agents according to the given boolean tensor.

        :param agents_to_reset: agents to reset of shape (batch_size, n_agents). A value
                                of 1 indicates that the agent's state should be reset.
        """
        if hasattr(self._model, "state") and self._model.state is not None:
            self._model.state = self._model.state * ~torch.tensor(
                agents_to_reset, dtype=bool, device=self._model.state.device
            ).unsqueeze(-1)

    def train(self):
        # switch back to old epsilon
        if self._epsilon_tmp is not None:
            self._epsilon_tmp = None
            self._epsilon = self._epsilon_tmp


class ShortestPath:
    def __init__(self, env, model, action_space, args) -> None:
        self._env = env
        assert isinstance(env.get(), Routing)
        self._n_agents = env.get_num_agents()
        self._model = model
        self._action_space = action_space
        self._args = args
        self.static_shortest_paths = False
        self.network = None

    def reset_episode(self):
        self.network = None

    def __call__(self, obs, adj):
        act = np.zeros(self._env.n_data, dtype=np.int32)

        if self.static_shortest_paths:
            # create shortest paths at the very beginning, then use them
            if self.network is None:
                import copy

                self.network = copy.deepcopy(self._env.network)

            network = self.network
        else:
            # always use latest shortest paths
            network = self._env.network

        for i in range(self._env.n_data):
            packet = self._env.data[i]
            current_node = packet.now
            target_node = packet.target

            if current_node == target_node:
                act[i] = 0
                continue

            # first index is the source and last index is the target
            next_node = network.shortest_paths[current_node][target_node][1]

            for index, j in enumerate(network.nodes[current_node].edges):
                current_edge = network.edges[j]
                # if current edge is the desired one we choose this edge
                # case 1: edge.start is the current node itself and edge.end is the target node
                if current_edge.get_other_node(packet.now) == next_node:
                    act[i] = index + 1
                    break

        return act


class RandomPolicy:
    def __init__(self, env, model, action_space, args) -> None:
        self._env = env
        self._model = model
        self._action_space = action_space
        self._args = args
        self._step = 0

    def __call__(self, obs, adj):
        self._step += 1
        self.action_space = Discrete(self._env.action_space.n, start=0)
        act = np.zeros(self._args.n_data, dtype=np.int32)
        # random action selection
        for i in range(len(act)):
            act[i] = self.action_space.sample()
        return act


class SimplePolicy:
    def __init__(self, env, model, action_space, args) -> None:
        self._env = env
        self._model = model
        self._action_space = action_space
        self._args = args
        self._step = 0

    def __call__(self, obs, adj):
        self._step += 1
        act = np.zeros(self._args.n_data, dtype=np.int32)
        edges = self._env.edges
        router = self._env.router

        for i in range(self._args.n_data):
            packet = self._env.data[i]
            current_node = packet.now
            target_node = router[0]

            if current_node == target_node:
                act[i] = 0
                continue
            else:
                for index, j in enumerate(router[current_node].edge):
                    current_edge = edges[j]
                    if current_edge.get_other_node(packet.now) == target_node:
                        act[i] = index + 1
                        break

        return act
