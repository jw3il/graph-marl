import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, AntiSymmetricConv, GraphSAGE
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn.summary import summary

# from torch_geometric_temporal import DyGrEncoder

from layernormlstm import LayerNormLSTMCell


class MLP(nn.Module):
    def __init__(
        self, in_features, mlp_units, activation_fn, activation_on_output=True
    ):
        super(MLP, self).__init__()
        self.activation_fn = activation_fn

        self.linear_layers = nn.ModuleList()
        previous_units = in_features
        if isinstance(mlp_units, int):
            mlp_units = [mlp_units]

        for units in mlp_units:
            self.linear_layers.append(nn.Linear(previous_units, units))
            previous_units = units

        self.out_features = previous_units
        self.activation_on_output = activation_on_output

    def forward(self, x):
        # intermediate layers
        for module in self.linear_layers[:-1]:
            x = self.activation_fn(module(x))

        # last layer
        x = self.linear_layers[-1](x)
        if self.activation_on_output:
            x = self.activation_fn(x)

        return x


class AttModel(nn.Module):
    """
    Multi-headed attention model based on..

    a) the following implementations of the paper "Graph Convolutional Reinforcement Learning"
    (https://arxiv.org/abs/1810.09202)
        a.1) ..in TensorFlow: https://github.com/PKU-RL/DGN
        a.2) ..in PyTorch: https://github.com/jiechuanjiang/pytorch_DGN/

    b) a PyTorch implementation of the Transformer model in "Attention is All You Need" (https://arxiv.org/abs/1706.03762)
        https://github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(
        self,
        in_features,
        k_features,
        v_features,
        out_features,
        num_heads,
        activation_fn,
        vkq_activation_fn,
    ):
        super(AttModel, self).__init__()
        self.k_features = k_features
        self.v_features = v_features
        self.num_heads = num_heads
        self.fc_v = nn.Linear(in_features, v_features * num_heads)
        self.fc_k = nn.Linear(in_features, k_features * num_heads)
        self.fc_q = nn.Linear(in_features, k_features * num_heads)
        self.fc_out = nn.Linear(v_features * num_heads, out_features)
        self.activation_fn = activation_fn
        self.vkq_activation_fn = vkq_activation_fn

        # attention scaling factor 1 / sqrt(d_k) from "Attention is All You Need"
        self.attention_scale = 1 / (k_features**0.5)

    def forward(self, x, mask):
        batch_size, num_agents = x.shape[0], x.shape[1]

        # get values, queries and keys and view according to heads
        # difference to DQN: we use a linear mapping here, as in the Transformer paper
        v = self.fc_v(x).view(batch_size, num_agents, self.num_heads, self.v_features)
        q = self.fc_q(x).view(batch_size, num_agents, self.num_heads, self.k_features)
        k = self.fc_k(x).view(batch_size, num_agents, self.num_heads, self.k_features)

        if self.vkq_activation_fn is not None:
            v = self.vkq_activation_fn(v)
            q = self.vkq_activation_fn(q)
            k = self.vkq_activation_fn(k)

        # permute for batch multiplication over batch size and heads
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # add head axis (mask is the same for all heads)
        mask = mask.unsqueeze(1)

        # calculate attention as dot product of all queries with all keys
        # according to mask and softmax over last dimension
        att_weights = torch.matmul(q, k.transpose(2, 3)) * self.attention_scale
        att = att_weights.masked_fill(mask == 0, -1e9)
        att = F.softmax(att, dim=-1)

        # combine values according to attention
        out = torch.matmul(att, v)
        # skip connection
        out = torch.add(out, v)
        # undo transpose and concatenate all heads
        out = out.transpose(1, 2).contiguous().view(batch_size, num_agents, -1)

        out = self.activation_fn(self.fc_out(out))
        return out, att_weights


class Q_Net(nn.Module):
    def __init__(self, in_features, actions):
        super(Q_Net, self).__init__()
        self.fc = nn.Linear(in_features, actions)

    def forward(self, x):
        return self.fc(x)


class DGN(nn.Module):
    def __init__(
        self,
        in_features,
        mlp_units,
        num_actions,
        num_heads,
        num_attention_layers,
        activation_fn,
    ):
        """
        Implementation of "Graph Convolutional Reinforcement Learning"
        (https://arxiv.org/abs/1810.09202) based on https://github.com/PKU-RL/DGN.

        :param in_features: Number of input features
        :param mlp_units: MLP units (either int or list/tuple)
        :param num_actions: Number of actions
        :param num_heads: Number of attention heads
        :param num_attention_layers: Number of attention layers
        :param activation_fn: activation function, defaults to F.relu
        """
        super(DGN, self).__init__()
        self.encoder = MLP(in_features, mlp_units, activation_fn)
        self.att_layers = nn.ModuleList()
        hidden_features = self.encoder.out_features
        for _ in range(num_attention_layers):
            self.att_layers.append(
                AttModel(
                    hidden_features,
                    # dv=16 from official implementation
                    # https://github.com/PKU-RL/DGN/blob/92b926888e82880afa68fcd967c6e6527f7773fa/Routing/routers.py#L196
                    16,
                    16,
                    hidden_features,
                    num_heads,
                    activation_fn,
                    # official implementation uses act function for key/query/value
                    activation_fn,
                )
            )

        self.q_net = Q_Net(hidden_features * (num_attention_layers + 1), num_actions)
        self.att_weights = []

    def forward(self, x, mask):
        h = self.encoder(x)

        q_input = h
        self.att_weights.clear()
        for attention_layer in self.att_layers:
            h, att_weights = attention_layer(h, mask)
            self.att_weights.append(att_weights)
            # concatenate outputs like described in the paper & official implementation
            q_input = torch.cat((q_input, h), dim=-1)

        q = self.q_net(q_input)
        return q


class DQN(nn.Module):
    """
    Minimal implementation of a DQN model (see https://www.nature.com/articles/nature14236)
    with vector-based input.
    """

    def __init__(self, in_features, mlp_units, num_actions, activation_fn):
        super(DQN, self).__init__()
        self.encoder = MLP(in_features, mlp_units, activation_fn)
        self.q_net = Q_Net(self.encoder.out_features, num_actions)
        self.activation_fn = activation_fn

    def forward(self, x, mask):
        batch, agent, features = x.shape
        h = self.encoder(x)
        q = self.q_net(h)
        return q


class SimpleAggregation(nn.Module):
    def __init__(self, agg: str, mask_eye: bool) -> None:
        super().__init__()
        self.agg = agg
        assert self.agg == "mean" or self.agg == "sum"
        self.mask_eye = mask_eye

    def forward(self, node_features, node_adjacency):
        if self.mask_eye:
            node_adjacency = node_adjacency * ~(
                torch.eye(
                    node_adjacency.shape[1],
                    node_adjacency.shape[1],
                    device=node_adjacency.device,
                )
                .repeat(node_adjacency.shape[0], 1, 1)
                .bool()
            )
        feature_sum = torch.bmm(node_adjacency, node_features)
        if self.agg == "sum":
            return feature_sum
        if self.agg == "mean":
            num_neighbors = torch.clamp(node_adjacency.sum(dim=-1), min=1).unsqueeze(-1)
            return feature_sum / num_neighbors


class JumpingKnowledgeADGN(nn.Module):
    """
    Stacks multiple iterations of AntiSymmetricConv (with weight sharing)
    and uses the provided jumping knowledge function to aggregate intermediate
    node states.
    """

    def __init__(self, hidden_features, num_iters, jk) -> None:
        super().__init__()
        self.aggregate = AntiSymmetricConv(hidden_features, num_iters=1)
        self.num_iters = num_iters
        assert self.num_iters >= 1
        self.jk = jk
        assert self.jk is not None

    def forward(self, x, mask_sparse):
        xs = []
        for _ in range(self.num_iters):
            x = self.aggregate(x, mask_sparse)
            xs.append(x)

        return self.jk(xs)


class NetMon(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features: int,
        encoder_units,
        iterations,
        activation_fn,
        rnn_type="lstm",
        rnn_carryover=True,
        agg_type="sum",
        output_neighbor_hidden=False,
        output_global_hidden=False,
    ) -> None:
        super().__init__()
        assert isinstance(hidden_features, int)
        self.encode = MLP(
            in_features,
            (*encoder_units, hidden_features),
            activation_fn,
        )
        self.state = None
        self.iterations = iterations
        self.output_neighbor_hidden = output_neighbor_hidden
        self.output_global_hidden = output_global_hidden
        self.rnn_carryover = rnn_carryover

        # 0 = dense input
        # 1 = sparse input
        # 2 = gconvlstm (sparse input)
        # 3 = GraphSAGE (sparse input, directly outputs neighbor info)
        self.aggregation_def_type = None

        # aggregation
        self.agg_type_str = agg_type
        # first resolve jumping knowledge functions for GraphSAGE and A-DGN
        self.jk = None
        if "jk-cat" in agg_type:
            self.jk_out = nn.Linear(hidden_features * iterations, hidden_features)
            self.jk_neighbors = nn.Linear(
                hidden_features * (iterations - 1), hidden_features
            )

            def jk_cat(xs):
                return (
                    self.jk_out(torch.cat(xs, dim=-1)),
                    self.jk_neighbors(torch.cat(xs[:-1], dim=-1)),
                )

            self.jk = jk_cat
        elif "jk-max" in agg_type:

            def jk_max(xs):
                return (
                    torch.max(torch.stack(xs), dim=0)[0],
                    torch.max(torch.stack(xs[:-1]), dim=0)[0],
                )

            self.jk = jk_max
        elif agg_type == "graphsage" or agg_type == "adgn":

            def jk(xs):
                return (xs[-1], xs[-2])

            self.jk = jk

        # now resolve the actual aggregation
        if agg_type == "sum" or agg_type == "mean":
            self.aggregate = SimpleAggregation(agg=agg_type, mask_eye=False)
            self.aggregation_def_type = 0
        elif agg_type == "gcn":
            self.aggregate = GCNConv(hidden_features, hidden_features, improved=True)
            self.aggregation_def_type = 1
        elif agg_type == "sage":
            self.aggregate = SAGEConv(hidden_features, hidden_features)
            self.aggregation_def_type = 1
        elif "graphsage" in agg_type:
            self.aggregate = GraphSAGE(
                hidden_features, hidden_features, num_layers=iterations
            )
            self.agg_type_str = agg_type + f" ({iterations} layer)"
            assert self.jk is not None
            self.aggregate.jk = self.jk
            self.aggregate.jk_mode = "custom"
            self.aggregation_def_type = 3
            self.iterations = 1
            if rnn_type != "none":
                print(f"WARNING: Overwritten given rnn type {rnn_type} with 'none'")
                rnn_type = "none"
        elif "adgn" in agg_type:
            self.aggregate = JumpingKnowledgeADGN(
                hidden_features, num_iters=iterations, jk=self.jk
            )
            self.agg_type_str = agg_type + f" ({iterations} layer)"
            self.aggregation_def_type = 3
            self.iterations = 1
            if rnn_type != "none":
                print(f"WARNING: Overwritten given rnn type {rnn_type} with 'none'")
                rnn_type = "none"
        elif agg_type == "antisymgcn":
            # use single iteration so that we still get last hidden node states
            self.aggregate = AntiSymmetricConv(hidden_features, num_iters=1)
            self.aggregation_def_type = 1
        elif agg_type == "gconvlstm":
            # filter size 1 => only from neighbors
            from torch_geometric_temporal.nn.recurrent.gconv_lstm import GConvLSTM

            self.agg_type_str = agg_type + f" (filter size {iterations + 1})"
            self.aggregate = GConvLSTM(
                hidden_features, hidden_features, K=(self.iterations + 1)
            )
            self.iterations = 1
            self.aggregation_def_type = 2
            if rnn_type != "gconvlstm":
                print(
                    f"WARNING: Overwritten given rnn type {rnn_type} with 'gconvlstm'"
                )
                rnn_type = "gconvlstm"
        else:
            raise ValueError(f"Unknown aggregation type {agg_type}")

        # update and observation encoding
        self.rnn_type = rnn_type
        if self.rnn_type == "lstm":
            self.rnn_obs = nn.LSTMCell(hidden_features, hidden_features)
            self.rnn_update = nn.LSTMCell(hidden_features, hidden_features)
            self.num_states = 2 if rnn_carryover else 4
        elif self.rnn_type == "lnlstm":
            self.rnn_obs = LayerNormLSTMCell(hidden_features, hidden_features)
            self.rnn_update = LayerNormLSTMCell(hidden_features, hidden_features)
            self.num_states = 2 if rnn_carryover else 4
        elif self.rnn_type == "gru":
            self.rnn_obs = nn.GRUCell(hidden_features, hidden_features)
            self.rnn_update = nn.GRUCell(hidden_features, hidden_features)
            self.num_states = 1 if rnn_carryover else 2
        elif self.rnn_type == "gconvlstm":
            # rnn is part of aggregate function
            self.num_states = 2
        elif self.rnn_type == "none":
            # empty state / stateless => simply store h for debugging
            self.num_states = 1
        else:
            raise ValueError(f"Unknown rnn type {self.rnn_type}")

        self.hidden_features = hidden_features
        self.state_size = hidden_features * self.num_states

    def get_out_features(self):
        out_features = self.hidden_features

        if self.output_neighbor_hidden:
            out_features += self.hidden_features * 3

        if self.output_global_hidden:
            out_features += self.hidden_features

        return out_features

    def get_state_size(self):
        return self.state_size

    def _state_reshape_in(self, batch_size, n_agents):
        """
        Reshapes the state of shape
            (batch_size, n_agents, self.get_state_len())
        to shape
            (2, batch_size * n_agents, hidden_size)

        :param batch_size: the batch size
        :param n_agents: the number of agents
        """
        if self.state.numel() == 0:
            return

        self.state = self.state.reshape(
            batch_size * n_agents,
            self.num_states,
            -1,
        ).transpose(0, 1)

    def _state_reshape_out(self, batch_size, n_agents):
        """
        Reshapes the state of shape
            (2, batch_size * n_agents, hidden_size)
        to shape
            (batch_size, n_agents, self.get_state_len()).

        :param batch_size: the batch size
        :param n_agents: the number of agents
        """
        if self.state.numel() == 0:
            return

        self.state = self.state.transpose(0, 1).reshape(batch_size, n_agents, -1)

    def forward(
        self, x, mask, node_agent_matrix, max_degree=None, no_agent_mapping=False
    ):
        # steps (1), (2) and (3)
        h, last_neighbor_h = self._update_node_states(x, mask)

        # step (4)
        if self.output_neighbor_hidden or self.output_global_hidden:
            extended_h = [h]

            if self.output_global_hidden:
                extended_h.append(self._get_global_h(h))

            if self.output_neighbor_hidden:
                extended_h.append(
                    self._get_neighbor_h(last_neighbor_h, mask, max_degree)
                )

            h = torch.cat(extended_h, dim=-1)

        if no_agent_mapping:
            return h

        return NetMon.output_to_network_obs(h, node_agent_matrix)

    def _update_node_states(self, x, mask):
        batch_size, n_nodes, feature_dim = x.shape
        x = x.reshape(batch_size * n_nodes, -1)

        if self.state is None:
            # initialize state
            self.state = torch.zeros(
                (batch_size, n_nodes, self.state_size), device=x.device
            )

        self._state_reshape_in(batch_size, n_nodes)

        # step (1): encode observation to get h^0_v and combine with state
        h = self.encode(x)
        if self.rnn_type == "lstm" or self.rnn_type == "lnlstm":
            h0, cx0 = self.rnn_obs(h, (self.state[0], self.state[1]))
            h, cx = h0, cx0
        elif self.rnn_type == "gru":
            h0 = self.rnn_obs(h, self.state[0])
            h = h0

        # message passing iterations
        if self.iterations <= 0 and self.output_neighbor_hidden:
            last_neighbor_h = torch.zeros_like(h, device=h.device)
        else:
            last_neighbor_h = None

        if self.aggregation_def_type != 0:
            mask_sparse, mask_weights = dense_to_sparse(mask)

        if self.aggregation_def_type == 2:
            H, C = self.state[0], self.state[1]

        for it in range(self.iterations):
            if self.output_neighbor_hidden and it == self.iterations - 1:
                if self.aggregation_def_type == 2:
                    # we know that the aggregation step will exchange the hidden states
                    # (and much more..) so we can just use them for the skip connection
                    # instead of the other nodes' input.
                    # This is only relevant for a single iteration per step.
                    last_neighbor_h = H
                else:
                    # use the last received hidden state
                    last_neighbor_h = h

            # step (2): aggregate
            if self.aggregation_def_type == 0:
                M = self.aggregate(h.view(batch_size, n_nodes, -1), mask).view(
                    batch_size * n_nodes, -1
                )
            elif self.aggregation_def_type == 1:
                M = self.aggregate(h, mask_sparse)
            elif self.aggregation_def_type == 2:
                H, C = self.aggregate(h, mask_sparse, H=H, C=C)
                M = H
            elif self.aggregation_def_type == 3:
                # overwrite last_neighbor_h with jumping knowledge output
                M, last_neighbor_h = self.aggregate(h, mask_sparse)

            # step (3): update
            # 23.03.23 GRU significantly better at regression task than simple update
            if self.rnn_type == "lstm" or self.rnn_type == "lnlstm":
                if not self.rnn_carryover and it == 0:
                    rnn_input = (self.state[2], self.state[3])
                else:
                    rnn_input = (h, cx)

                h1, cx1 = self.rnn_update(M, rnn_input)
                h, cx = h1, cx1
            elif self.rnn_type == "gru":
                if not self.rnn_carryover and it == 0:
                    rnn_input = self.state[1]
                else:
                    rnn_input = h

                h1 = self.rnn_update(M, rnn_input)
                h = h1
            else:
                h = M

        # reshape
        if last_neighbor_h is not None:
            last_neighbor_h = last_neighbor_h.reshape(batch_size, n_nodes, -1)
        h = h.reshape(batch_size, n_nodes, -1)

        # update internal state
        if self.rnn_type == "lstm" or self.rnn_type == "lnlstm":
            if self.rnn_carryover:
                self.state = torch.stack((h1, cx1))
            else:
                self.state = torch.stack((h0, cx0, h1, cx1))
        elif self.rnn_type == "gru":
            if self.rnn_carryover:
                self.state = h1.unsqueeze(0)
            else:
                self.state = torch.stack((h0.unsqueeze(0), h1.unsqueeze(0)))
        elif self.rnn_type == "gconvlstm":
            self.state = torch.stack((H, C))
        elif self.rnn_type == "none":
            # store last node state for debugging and aux loss
            self.state = h.unsqueeze(0)

        self._state_reshape_out(batch_size, n_nodes)

        return h, last_neighbor_h

    def _get_neighbor_h(self, neighbor_h, mask, max_degree):
        batch_size, n_nodes, _ = neighbor_h.shape
        # return own hidden state + last received neighbor hidden states ordered
        # by node ids

        # get max node id for dense observation tensor (excluding self)
        if max_degree is None:
            max_degree = torch.sum(mask, dim=-1).max().long().item() - 1

        # placeholder for observations for each neighbor
        h_neighbors = torch.zeros(
            (batch_size, n_nodes, max_degree, neighbor_h.shape[-1]),
            device=neighbor_h.device,
        )

        # get mask without self (only containing neighbors)
        neighbor_mask = mask * ~(
            torch.eye(n_nodes, n_nodes, device=mask.device)
            .unsqueeze(0)
            .repeat(mask.shape[0], 1, 1)
            .bool()
        )

        # we want to collect features from neighbors and put them into h_neighbors
        # 1) get the neighbor node indices (batch, node, neighbor)
        h_index = neighbor_mask.nonzero()

        # 2) get the relative neighbor id for the insertion in h_neighbors
        # first neighbor (with lowest node id) is neighbor 0, then the ids increase
        cumulative_neighbor_index = neighbor_mask.cumsum(dim=-1).long() - 1
        h_neighbors_index = cumulative_neighbor_index[
            h_index[:, 0], h_index[:, 1], h_index[:, 2]
        ]

        # 3) copy the last hidden states of all neighbors into the h_neighbors tensor
        h_neighbors[h_index[:, 0], h_index[:, 1], h_neighbors_index] = neighbor_h[
            h_index[:, 0], h_index[:, 2]
        ]

        # concatenate info for each node
        return h_neighbors.reshape(batch_size, n_nodes, -1)

    def _get_global_h(self, h):
        _, n_nodes, _ = h.shape
        global_h = h.mean(dim=1).repeat((n_nodes, 1, 1)).transpose(0, 1)
        return global_h

    @staticmethod
    def output_to_network_obs(netmon_out, node_agent_matrix):
        return torch.bmm(netmon_out.transpose(1, 2), node_agent_matrix).transpose(1, 2)

    def summarize(self, *args):
        str_out = []
        str_out.append("NetMon Module")
        str_out.append(summary(self, *args, max_depth=10))
        self.state = None
        str_out.append(f"> Aggregation Type: {self.agg_type_str}")
        str_out.append(f"> RNN Type: {self.rnn_type}")
        str_out.append(f"> Carryover: {self.rnn_carryover}")
        str_out.append(f"> Iterations: {self.iterations}")
        readout_str = "> Readout: local"
        if self.output_neighbor_hidden:
            readout_str += " + last neighbors"
        if self.output_global_hidden:
            readout_str += " + global agg"
        str_out.append(readout_str)
        import os

        return os.linesep.join(str_out)


class DQNR(nn.Module):
    """
    Recurrent DQN with an lstm cell.
    """

    def __init__(self, in_features, mlp_units, num_actions, activation_fn):
        super(DQNR, self).__init__()
        self.encoder = MLP(in_features, mlp_units, activation_fn)
        self.lstm = nn.LSTMCell(
            input_size=self.encoder.out_features, hidden_size=self.encoder.out_features
        )
        self.state = None
        self.q_net = Q_Net(self.encoder.out_features, num_actions)

    def get_state_len(self):
        return 2 * self.lstm.hidden_size

    def _state_reshape_in(self, batch_size, n_agents):
        """
        Reshapes the state of shape
            (batch_size, n_agents, self.get_state_len())
        to shape
            (2, batch_size * n_agents, hidden_size).

        :param batch_size: the batch size
        :param n_agents: the number of agents
        """
        self.state = (
            self.state.reshape(
                batch_size * n_agents,
                2,
                self.lstm.hidden_size,
            )
            .transpose(0, 1)
            .contiguous()
        )

    def _state_reshape_out(self, batch_size, n_agents):
        """
        Reshapes the state of shape
            (2, batch_size * n_agents, hidden_size)
        to shape
            (batch_size, n_agents, self.get_state_len()).

        :param batch_size: the batch size
        :param n_agents: the number of agents
        """
        self.state = self.state.transpose(0, 1).reshape(batch_size, n_agents, -1)

    def _lstm_forward(self, x, reshape_state=True):
        """
        A single lstm forward pass

        :param x: Cell input
        :param reshape_state: reshape the state to and from (batch_size, n_agents, -1)
        """
        batch_size, n_agents, feature_dim = x.shape
        # combine agent and batch dimension
        x = x.view(batch_size * n_agents, -1)

        if self.state is None:
            lstm_hidden_state, lstm_cell_state = self.lstm(x)
        else:
            if reshape_state:
                self._state_reshape_in(batch_size, n_agents)
            lstm_hidden_state, lstm_cell_state = self.lstm(
                x, (self.state[0], self.state[1])
            )

        self.state = torch.stack((lstm_hidden_state, lstm_cell_state))
        x = lstm_hidden_state

        # undo combine
        x = x.view(batch_size, n_agents, -1)
        if reshape_state:
            self._state_reshape_out(batch_size, n_agents)

        return x

    def forward(self, x, mask):
        h = self.encoder(x)
        h = self._lstm_forward(h)
        return self.q_net(h)


class CommNet(DQNR):
    """
    Implementation of CommNet https://arxiv.org/abs/1605.07736 with masked communication
    between agents.

    While the hidden state is aggregated over the neighbors during communication, the
    individual cell states stay the same. This is how IC3Net implemented CommNet
    https://github.com/IC3Net/IC3Net. The CommNet paper does not elaborate on if and how
    the cell states are combined.
    """

    def __init__(
        self,
        in_features,
        mlp_units,
        num_actions,
        comm_rounds,
        activation_fn,
    ):
        super().__init__(in_features, mlp_units, num_actions, activation_fn)
        assert comm_rounds >= 0
        self.comm_rounds = comm_rounds

    def forward(self, x, mask):
        batch_size, n_agents, feature_dim = x.shape
        h = self.encoder(x)

        # manually reshape state
        if self.state is not None:
            self._state_reshape_in(batch_size, n_agents)

        h = self._lstm_forward(h, reshape_state=False)

        # explicitly exclude self-communication from mask
        mask = mask * ~torch.eye(n_agents, dtype=bool, device=x.device).unsqueeze(0)

        for _ in range(self.comm_rounds):
            # combine hidden state h according to mask
            # first add up hidden states according to mask
            #    h has dimensions (batch, agents, features)
            #    and mask has dimensions (batch, agents, neighbors)
            #    => we have to transpose the mask to aggregate over all neighbors
            c = torch.bmm(h.transpose(1, 2), mask.transpose(1, 2)).transpose(1, 2)
            # then normalize according to number of neighbors per agent
            c = c / torch.clamp(mask.sum(dim=-1).unsqueeze(-1), min=1)

            # skip connection for hidden state and communication
            h = h + c
            # use new hidden state
            self.state[0] = h.view(batch_size * n_agents, -1)

            # pass through forward module
            h = self._lstm_forward(h, reshape_state=False)

        # manually reshape state in the end
        self._state_reshape_out(batch_size, n_agents)
        return self.q_net(h)
