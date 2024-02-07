from collections import namedtuple
from typing import Iterator
import numpy as np
import matplotlib.pyplot as plt

import torch

# named tuple for easier access of batches
TransitionBatch = namedtuple(
    "TransitionBatch",
    [
        "idx",
        "obs",
        "action",
        "reward",
        "next_obs",
        "adj",
        "next_adj",
        "done",
        "episode_done",
        "agent_state",
        "node_obs",
        "node_adj",
        "node_state",
        "node_aux",
        "node_agent_matrix",
        "next_node_obs",
        "next_node_adj",
        "next_node_agent_matrix",
    ],
)


class ReplayBuffer(object):
    def __init__(
        self,
        seed,
        buffer_size,
        n_agents,
        observation_size,
        agent_state_size,
        n_nodes=0,
        node_observation_size=0,
        node_state_size=0,
        node_aux_size=0,
        half_precision=False,
    ):
        self.buffer_size = buffer_size
        self.count = 0
        self.index = 0

        # create buffer

        def create(shape, dtype, default=0):
            act = np.empty(shape, dtype=dtype)
            act.fill(default)
            return act

        if half_precision:
            float_type = np.float16
        else:
            float_type = np.float32

        self.obs = create((buffer_size, n_agents, observation_size), dtype=float_type)
        self.action = create((buffer_size, n_agents), dtype=np.int8)
        self.reward = create((buffer_size, n_agents), dtype=float_type)
        self.next_obs = create(
            (buffer_size, n_agents, observation_size), dtype=float_type
        )
        self.adj = create((buffer_size, n_agents, n_agents), dtype=np.bool_)
        self.next_adj = create((buffer_size, n_agents, n_agents), dtype=np.bool_)
        self.done = create((buffer_size, n_agents), dtype=np.bool_)
        self.episode_done = create(buffer_size, dtype=np.bool_)
        self.agent_state = create(
            (buffer_size, n_agents, agent_state_size), dtype=float_type
        )

        # optional node information
        if n_nodes > 0:
            assert node_observation_size >= 0 and node_state_size >= 0

        self.node_state = create(
            (buffer_size, n_nodes, node_state_size), dtype=float_type
        )
        self.node_aux = create((buffer_size, n_nodes, node_aux_size), dtype=float_type)
        self.node_obs = create(
            (buffer_size, n_nodes, node_observation_size), dtype=float_type
        )
        self.next_node_obs = create(
            (buffer_size, n_nodes, node_observation_size), dtype=float_type
        )
        self.node_adj = create((buffer_size, n_nodes, n_nodes), dtype=np.bool_)
        self.next_node_adj = create((buffer_size, n_nodes, n_nodes), dtype=np.bool_)
        self.node_agent_matrix = create(
            (buffer_size, n_nodes, n_agents), dtype=np.bool_
        )
        self.next_node_agent_matrix = create(
            (buffer_size, n_nodes, n_agents), dtype=np.bool_
        )

        self._random_generator = np.random.default_rng(seed)

    def get_batch(
        self, batch_size, device, sequence_length=0
    ) -> Iterator[TransitionBatch]:
        # simple case: just get random indices
        if sequence_length <= 1:
            indices = self._random_generator.choice(
                self.count, batch_size, replace=True, p=None
            )
            yield self._get_transition_batch(indices, device)
            return

        # we sample a sequence with length > 1

        # first get the beginning of the buffer (oldest element)
        buffer_start = self.index % self.count
        batch_sequence_start = self._random_generator.choice(
            self.count - sequence_length,
            batch_size,
            replace=True,
            p=None,
        )

        # add buffer start offset and wrap around
        batch_sequence_start = (buffer_start + batch_sequence_start) % self.count

        for offset in range(sequence_length):
            indices = (batch_sequence_start + offset) % self.count
            yield self._get_transition_batch(indices, device)

    def _get_transition_batch(self, indices, device) -> TransitionBatch:
        # convert to tensor and push to training device
        return TransitionBatch(
            indices,
            torch.tensor(self.obs[indices], dtype=torch.float32).to(
                device, non_blocking=True
            ),
            torch.tensor(self.action[indices], dtype=torch.int64).to(
                device, non_blocking=True
            ),
            torch.tensor(self.reward[indices], dtype=torch.float32).to(
                device, non_blocking=True
            ),
            torch.tensor(self.next_obs[indices], dtype=torch.float32).to(
                device, non_blocking=True
            ),
            torch.tensor(self.adj[indices], dtype=torch.float32).to(
                device, non_blocking=True
            ),
            torch.tensor(self.next_adj[indices], dtype=torch.float32).to(
                device, non_blocking=True
            ),
            torch.tensor(self.done[indices], dtype=torch.bool).to(
                device, non_blocking=True
            ),
            torch.tensor(self.episode_done[indices], dtype=torch.bool).to(
                device, non_blocking=True
            ),
            torch.tensor(self.agent_state[indices], dtype=torch.float32).to(
                device, non_blocking=True
            ),
            torch.tensor(self.node_obs[indices], dtype=torch.float32).to(
                device, non_blocking=True
            ),
            torch.tensor(self.node_adj[indices], dtype=torch.float32).to(
                device, non_blocking=True
            ),
            torch.tensor(self.node_state[indices], dtype=torch.float32).to(
                device, non_blocking=True
            ),
            torch.tensor(self.node_aux[indices], dtype=torch.float32).to(
                device, non_blocking=True
            ),
            torch.tensor(self.node_agent_matrix[indices], dtype=torch.float32).to(
                device, non_blocking=True
            ),
            torch.tensor(self.next_node_obs[indices], dtype=torch.float32).to(
                device, non_blocking=True
            ),
            torch.tensor(self.next_node_adj[indices], dtype=torch.float32).to(
                device, non_blocking=True
            ),
            torch.tensor(self.next_node_agent_matrix[indices], dtype=torch.float32).to(
                device, non_blocking=True
            ),
        )

    def get_recent_indices(self, last_n):
        if last_n is None:
            return np.arange(self.count), np.arange(self.count)

        m = min(self.count, last_n)
        x = self.index - m + np.arange(m)
        return x, x % self.count

    def plot_episode_vlines(self, idx, ymin, ymax):
        episode_done = self.episode_done[idx].nonzero()[0]
        plt.vlines(
            idx[0] + episode_done,
            colors="k",
            linestyles="dotted",
            ymin=ymin,
            ymax=ymax,
        )

    def save_node_state_diff_plot(self, filename, last_n):
        # only for debugging
        x, idx = self.get_recent_indices(last_n)
        node_state_diff = (self.node_state[idx[1:]] - self.node_state[idx[:-1]]).mean(
            axis=(-1, -2)
        )
        self.plot_episode_vlines(idx[1:], node_state_diff.min(), node_state_diff.max())
        plt.plot(x[1:], node_state_diff)
        plt.xlabel("Buffer steps")
        plt.ylabel("Node state difference")
        plt.tight_layout()
        plt.savefig(
            filename,
            bbox_inches="tight",
        )
        plt.clf()

    def save_node_state_std_plot(self, filename, last_n):
        # std over node dim => std per feature
        x, idx = self.get_recent_indices(last_n)
        node_states_feature_std = self.node_state[idx].std(axis=-2)
        mean = node_states_feature_std.mean(axis=-1)
        std = node_states_feature_std.std(axis=-1)
        self.plot_episode_vlines(x, (mean - std).min(), (mean + std).max())
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        plt.plot(x, mean)
        plt.xlabel("Buffer steps")
        plt.ylabel("Node feature std")

        plt.tight_layout()
        plt.savefig(
            filename,
            bbox_inches="tight",
        )
        plt.clf()

    def add(
        self,
        obs,
        action,
        reward,
        next_obs,
        adj,
        next_adj,
        done,
        episode_done,
        agent_state,
        node_state,
        node_aux,
        node_obs,
        node_adj,
        node_agent_matrix,
        next_node_obs,
        next_node_adj,
        next_node_agent_matrix,
    ):
        # put values into buffer
        self.obs[self.index] = obs
        self.action[self.index] = action
        self.reward[self.index] = reward
        self.next_obs[self.index] = next_obs
        self.adj[self.index] = adj
        self.next_adj[self.index] = next_adj
        self.done[self.index] = done
        self.episode_done[self.index] = episode_done
        if isinstance(agent_state, np.ndarray):
            agent_state = np.squeeze(agent_state, axis=0)
        self.agent_state[self.index] = agent_state
        self.node_state[self.index] = node_state
        self.node_aux[self.index] = node_aux
        self.node_obs[self.index] = node_obs
        self.node_adj[self.index] = node_adj
        self.node_agent_matrix[self.index] = node_agent_matrix
        self.next_node_obs[self.index] = next_node_obs
        self.next_node_adj[self.index] = next_node_adj
        self.next_node_agent_matrix[self.index] = next_node_agent_matrix

        # increase counters
        if self.count < self.buffer_size:
            self.count += 1
        self.index = (self.index + 1) % self.buffer_size
