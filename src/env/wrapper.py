import os
import torch
import numpy as np
from model import NetMon


class NetMonWrapper:
    """
    Wraps a given environment with a netmon instance. Creates new observations by
    concatenating the agent's observations and the respective graph observations.
    """

    def __init__(self, env, netmon, startup_iterations) -> None:
        self.env = env
        self.netmon = netmon
        self.device = next(netmon.parameters()).device

        self.node_obs = None
        self.node_adj = None
        self.node_agent_matrix = None
        self.last_netmon_state = None
        self.current_netmon_state = None
        assert startup_iterations >= 1, "Number of startup iterations must be >= 1"
        self.startup_iterations = startup_iterations
        self.frozen = False

    def __getattr__(self, name):
        # allow to access attributes of the environment
        return getattr(self.env, name)

    def __str__(self) -> str:
        return (
            self.env.__str__()
            + os.linesep
            + "▲ environment is wrapped with NetMon (graph obs)"
        )

    def reset(self):
        self.frozen = False
        self.current_netmon_state = None
        self.last_netmon_state = None
        obs, adj = self.env.reset()
        for _ in range(self.startup_iterations):
            network_obs = self._netmon_step()
        return np.concatenate((obs, network_obs), axis=-1), adj

    def step(self, actions):
        next_obs, next_adj, reward, done, info = self.env.step(actions)
        next_network_obs = self._netmon_step()
        next_joint_obs = np.concatenate((next_obs, next_network_obs), axis=-1)
        return next_joint_obs, next_adj, reward, done, info

    def freeze(self):
        """
        Disable message-passing for the rest of this episode. Agents still receive
        graph observations depending on their position in the graph.
        """
        self.frozen = True

    def get_netmon_info(self):
        return (self.node_obs, self.node_adj, self.node_agent_matrix)

    def get(self):
        return self.env

    def _netmon_step(self):
        if self.frozen:
            self.node_agent_matrix = self.env.get_node_agent_matrix()
            node_agent_matrix_in = torch.tensor(
                self.node_agent_matrix, dtype=torch.float32
            ).unsqueeze(0)
            network_obs = torch.bmm(
                self.netmon_out.transpose(1, 2).cpu().detach(), node_agent_matrix_in
            ).transpose(1, 2)
            return network_obs.squeeze(0).numpy()

        self.node_obs = self.env.get_node_observation()
        self.node_adj = self.env.get_nodes_adjacency()
        self.node_agent_matrix = self.env.get_node_agent_matrix()
        with torch.no_grad():
            # prepare inputs
            node_obs_in = (
                torch.tensor(self.node_obs, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device, non_blocking=True)
            )
            node_adj_in = (
                torch.tensor(self.node_adj, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device, non_blocking=True)
            )
            node_agent_matrix_in = (
                torch.tensor(self.node_agent_matrix, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device, non_blocking=True)
            )

            # perform netmon step with correct state
            self.last_netmon_state = self.current_netmon_state
            self.netmon.state = self.current_netmon_state
            self.netmon_out = self.netmon(
                node_obs_in, node_adj_in, node_agent_matrix_in, no_agent_mapping=True
            )
            self.current_netmon_state = self.netmon.state

            network_obs = NetMon.output_to_network_obs(
                self.netmon_out, node_agent_matrix_in
            )
            return network_obs.squeeze(0).cpu().detach().numpy()
