import argparse

from pathlib import Path
import pickle
from tqdm import tqdm
from env.constants import EVAL_SEEDS
from env.network import Network
from model import NetMon
from env.environment import EnvironmentVariant
from env.routing import Routing
import torch.nn as nn
import torch
import numpy as np
import networkx as nx
import pandas as pd
import torch.nn.functional as F

from util import dim_str_to_list, set_seed

parser = argparse.ArgumentParser(
    description="Train and test graph observation models on a supervised routing task."
)

parser.add_argument(
    "--num-targets",
    type=int,
    help="Number of targets included in the loss for regression with all destinations",
    default=None,
)
parser.add_argument(
    "--num-samples-train",
    type=int,
    help="Number of generated training graphs (ignored when loading a dataset)",
    default=10_000,
)
parser.add_argument("--seed", type=int, help="Seed for the experiment", default=42)
parser.add_argument(
    "--iterations", type=int, help="Number of training iterations", default=4_000
)
parser.add_argument(
    "--validate-after",
    type=int,
    help="Validate model after the given number of training steps",
    default=1_000,
)
parser.add_argument(
    "--sequence-length",
    type=int,
    help="Unroll depth of the model for each sample",
    default=4,
)
parser.add_argument(
    "--filename", type=str, help="Where to save the results", default=None
)
parser.add_argument(
    "--test-sequence-lengths",
    type=str,
    help="Sequence lengths used during testing",
    default="1,2,4,8,16,32,64,128,256",
)
parser.add_argument(
    "--netmon-dim", type=int, help="Size of NetMon state and observations", default=128
)
parser.add_argument(
    "--netmon-encoder-dim",
    type=str,
    help="NetMon encoder dimensions. Examples: '128', '512,128'..",
    default="512,256",
)
parser.add_argument(
    "--netmon-iterations",
    type=int,
    help="Number of NetMon iterations between environment steps",
    default=3,
)
parser.add_argument(
    "--netmon-rnn-type",
    type=str,
    help="NetMon RNN type",
    default="lstm",
)
parser.add_argument(
    "--netmon-rnn-carryover",
    type=int,
    help="Carry over RNN state between RNN modules",
    # 0: False, 1: True
    choices=[False, True],
    default=True,
)
parser.add_argument(
    "--netmon-agg-type",
    type=str,
    help="NetMon aggregation function",
    default="sum",
)
parser.add_argument(
    "--netmon-global",
    help="Enables global pooling of graph observations (only allowed in centralized case)",
    dest="netmon_global",
    action="store_true",
)
parser.set_defaults(netmon_global=False)
parser.add_argument(
    "--netmon-last-neighbors",
    type=int,
    help="Append last node state received by neighbors to graph observation",
    choices=[False, True],
    default=True,
)

parser.add_argument(
    "--disable-progressbar",
    help="Disables the progress bar and iteration-wise status prints",
    dest="disable_progressbar",
    action="store_true",
)
parser.set_defaults(disable_progressbar=False)

parser.add_argument(
    "--clear-cache",
    help="Forces generation of new datasets (clears cache if it exists)",
    dest="clear_cache",
    action="store_true",
)
parser.set_defaults(clear_cache=False)

args = parser.parse_args()
args.test_sequence_lengths = dim_str_to_list(args.test_sequence_lengths)
set_seed(args.seed)


class NetMonSL(nn.Module):
    def __init__(self, node_obs_dim, nb_classes, nb_nodes) -> None:
        super().__init__()
        # rnn_type specifies if we are using LSTM or GRU
        self.netmon = NetMon(
            node_obs_dim,
            args.netmon_dim,
            dim_str_to_list(args.netmon_encoder_dim),
            iterations=args.netmon_iterations,
            activation_fn=F.leaky_relu,
            rnn_type=args.netmon_rnn_type,
            rnn_carryover=args.netmon_rnn_carryover,
            agg_type=args.netmon_agg_type,
            output_neighbor_hidden=args.netmon_last_neighbors,
            output_global_hidden=args.netmon_global,
        )
        self.linear = nn.Linear(self.netmon.get_out_features(), nb_classes)
        self.linear_reg = nn.Linear(self.netmon.get_out_features(), 1)
        self.linear_reg_all = nn.Linear(self.netmon.get_out_features(), nb_nodes)
        self.class_logits = None

    def forward(self, node_obs, node_adj):
        batches, nodes, features = node_obs.shape
        eye = torch.eye(nodes).repeat(batches, 1, 1)

        node_features = self.netmon(node_obs, node_adj, eye)
        class_logits = self.linear(node_features)
        pred = self.linear_reg(node_features)
        pred_all = self.linear_reg_all(node_features)
        self.class_logits = class_logits.detach()
        return class_logits, pred, pred_all

    def get_class_probabilities(self):
        return torch.softmax(self.class_logits, dim=-1)

    def get_prediction(self):
        return torch.argmax(self.get_class_probabilities(), axis=-1)


NUM_CLASSES = 4


def get_sl_sample(env: Routing):
    single_node_obs = env.get_node_observation()
    single_node_adj = env.get_nodes_adjacency()
    single_node_labels = np.zeros(single_node_obs.shape[0])
    single_node_targets = np.zeros(single_node_obs.shape[0])
    single_node_targets_all = np.zeros(
        (single_node_obs.shape[0], single_node_obs.shape[0])
    )

    # get labels from path to zero
    for n in range(env.get_num_nodes()):
        # distance to node 0
        single_node_targets[n] = env.network.shortest_paths_weights[n][0]
        for n_other in range(env.get_num_nodes()):
            single_node_targets_all[n, n_other] = env.network.shortest_paths_weights[n][
                n_other
            ]

        # which link corresponds to the shortest path to node 0?
        n_to_zero = env.network.shortest_paths[n][0]
        if len(n_to_zero) == 1:
            # we are already there
            single_node_labels[n] = 0
            # print(f"{n}: is the target")
        else:
            # look for the link we need to take
            next_node = n_to_zero[1]
            found_edge = False
            for e_idx, e in enumerate(env.network.nodes[n].edges):
                if env.network.edges[e].get_other_node(n) == next_node:
                    single_node_labels[n] = e_idx + 1
                    # print(f"{n}: found node {next_node} at edge idx {e_idx + 1}")
                    found_edge = True
                    break

            assert found_edge

    assert (single_node_adj.sum(axis=-1) == 4).all()
    return (
        single_node_obs,
        single_node_adj,
        single_node_labels,
        single_node_targets,
        single_node_targets_all,
    )


# we are getting the average steps from one node to node 0
def get_mean_num_shortest_paths_to_zero(env):
    num_paths = []
    for i in range(env.get_num_nodes()):
        num_paths.append(
            len(
                list(
                    nx.all_shortest_paths(
                        env.network.G, i, 0, weight=env.network.G_weight_key
                    )
                )
            )
        )
    return np.mean(num_paths)


def build_dataset(env: Routing, num_samples):
    assert num_samples >= 1
    (
        init_node_obs,
        init_node_adj,
        init_node_labels,
        init_node_targets,
        init_node_targets_all,
    ) = get_sl_sample(env)

    node_obs = np.zeros((num_samples, *init_node_obs.shape))
    node_adj = np.zeros((num_samples, *init_node_adj.shape))
    node_labels = np.zeros((num_samples, *init_node_labels.shape))
    node_targets = np.zeros((num_samples, *init_node_targets.shape))
    node_targets_all = np.zeros((num_samples, *init_node_targets_all.shape))

    node_obs[0] = init_node_obs
    node_adj[0] = init_node_adj
    node_labels[0] = init_node_labels
    node_targets[0] = init_node_targets
    node_targets_all[0] = init_node_targets_all

    mean_paths = np.zeros((num_samples))
    mean_paths[0] = get_mean_num_shortest_paths_to_zero(env)

    for s in tqdm(
        range(1, num_samples),
        initial=1,
        total=num_samples,
        disable=args.disable_progressbar,
    ):
        env.reset()
        (
            node_obs[s],
            node_adj[s],
            node_labels[s],
            node_targets[s],
            node_targets_all[s],
        ) = get_sl_sample(env)
        mean_paths[s] = get_mean_num_shortest_paths_to_zero(env)

    print(
        "Network stats: \n"
        f"Mean neighbors: {node_adj.sum(axis=-1).mean()} \n"
        f"Mean distance to 0: {node_targets.mean()} \n"
        f"Max distance to 0: {node_targets.max()} \n"
        f"Mean shortest paths to 0: {mean_paths.mean()}"
    )

    return node_obs, node_adj, node_labels, node_targets, node_targets_all


def build_or_load_dataset(env: Routing, num_samples, filename, clear_cache, cache):
    path = Path(filename)

    if path.exists():
        if clear_cache:
            path.unlink()
        elif cache:
            with open(path, "rb") as f:
                dataset = pickle.load(f)
                return dataset

    print("Creating new dataset..")
    dataset = build_dataset(env, num_samples)

    if cache:
        with open(path, "wb") as f:
            pickle.dump(dataset, f)
            print(f"Saved dataset as {path}")

    return dataset


class RandomLossReduction:
    def __init__(self, loss) -> None:
        self.loss = loss

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_out = self.loss(input, target)
        loss_weights = torch.ones_like(loss_out)
        loss_weights *= torch.rand(size=loss_out.shape) >= 0.5
        return torch.sum(loss_out * loss_weights) / torch.sum(loss_weights)


def train(
    model: nn.Module,
    train_node_obs,
    train_node_adj,
    train_node_labels,
    train_node_targets,
    train_node_targets_all,
    iterations,
    batch_size,
    with_classification,
    with_regression,
    with_regression_all,
    sequence_length,
    random_loss_weights=False,
    validation_callback=None,
):
    assert len(train_node_obs) == len(train_node_adj) == len(train_node_labels)
    num_samples = len(train_node_obs)
    optim = torch.optim.AdamW(model.parameters())

    reduction = "none" if random_loss_weights else "mean"
    cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction)
    mse = torch.nn.MSELoss(reduction=reduction)

    if random_loss_weights:
        cross_entropy = RandomLossReduction(cross_entropy)
        mse = RandomLossReduction(mse)

    model.train()
    class_loss_list = []
    reg_loss_list = []
    reg_all_loss_list = []
    total_loss_list = []
    validation_list = []

    if validation_callback is not None:
        tqdm.write("Validation..")
        netmon.eval()
        validation_list.append((0, *validation_callback()))
        netmon.train()

    for it in tqdm(
        range(iterations), total=iterations, disable=args.disable_progressbar
    ):
        sequence_loss_list = []
        model.netmon.state = None
        batch_idx = np.random.choice(num_samples, batch_size, replace=True)
        batch_node_obs = torch.Tensor(train_node_obs[batch_idx])
        batch_node_adj = torch.Tensor(train_node_adj[batch_idx])
        batch_node_labels = torch.Tensor(train_node_labels[batch_idx]).long()
        batch_node_targets = torch.Tensor(train_node_targets[batch_idx])
        batch_node_targets_all = torch.Tensor(train_node_targets_all[batch_idx])

        for _ in range(0, max(sequence_length, 1)):
            log_probs, pred, pred_all = model(batch_node_obs, batch_node_adj)

            loss = 0
            if with_classification:
                # combine batch and node dimensions for loss calculation
                log_probs_loss = log_probs.reshape(-1, NUM_CLASSES)
                batch_node_labels_loss = batch_node_labels.reshape(-1, 1)

                class_loss = cross_entropy(
                    log_probs_loss, batch_node_labels_loss.squeeze(-1)
                )
                loss += class_loss

            if with_regression:
                pred = pred.reshape(-1, 1)
                batch_node_targets = batch_node_targets.reshape(-1, 1)
                regression_loss = mse(pred, batch_node_targets)
                loss += regression_loss

            if with_regression_all:
                regression_loss_all = mse(
                    pred_all[..., : args.num_targets],
                    batch_node_targets_all[..., : args.num_targets],
                )
                loss += regression_loss_all

            # remember total loss for each element in the sequence
            sequence_loss_list.append(loss)

        # log loss at end of sequence
        iteration_str = f"Iteration {it}"
        if with_classification:
            class_loss_list.append(class_loss.detach().item())
            iteration_str += f" | class loss = {class_loss.detach().item():.2f}"
        if with_regression:
            reg_loss_list.append(regression_loss.detach().item())
            iteration_str += f" | reg loss = {regression_loss.detach().item():.2f}"
        if with_regression_all:
            reg_all_loss_list.append(regression_loss_all.detach().item())
            iteration_str += (
                f" | reg_all loss = {regression_loss_all.detach().item():.2f}"
            )

        # and total mean loss for the sequence
        total_loss = torch.mean(torch.stack(sequence_loss_list))
        total_loss_list.append(total_loss.detach().item())
        iteration_str += f" | total = {total_loss.detach().item():.2f}"
        if sequence_length > 1:
            iteration_str += f" (seq_len={sequence_length})"

        optim.zero_grad()
        total_loss.backward()
        optim.step()

        if not args.disable_progressbar:
            tqdm.write(iteration_str)

        if validation_callback is not None and (it + 1) % args.validate_after == 0:
            tqdm.write(f"Iteration {it + 1}: validation")
            netmon.eval()
            validation_list.append((it + 1, *validation_callback()))
            netmon.train()

    return (
        total_loss_list,
        class_loss_list,
        reg_loss_list,
        reg_all_loss_list,
        validation_list,
    )


def test(
    model: nn.Module,
    test_node_obs,
    test_node_adj,
    test_node_labels,
    test_node_targets,
    test_node_targets_all,
    batch_size,
    with_classification,
    with_regression,
    with_regression_all,
    sequence_length,
    disable_progress_force=False,
):
    assert len(test_node_obs) == len(test_node_adj) == len(test_node_labels)
    num_samples = len(test_node_obs)

    cross_entropy = torch.nn.CrossEntropyLoss(reduction="sum")
    mse = torch.nn.MSELoss(reduction="sum")

    total_class_loss = 0
    total_reg_loss = 0
    total_reg_all_loss = 0
    total_correct = 0
    total_count = np.prod(test_node_obs.shape[0:2])

    model.eval()
    idx = 0
    pbar = tqdm(
        total=num_samples, disable=args.disable_progressbar or disable_progress_force
    )
    with torch.no_grad():
        while idx < num_samples:
            model.netmon.state = None
            next_idx = min(num_samples, idx + batch_size)
            batch_idx = np.arange(idx, next_idx)
            batch_node_obs = torch.Tensor(test_node_obs[batch_idx])
            batch_node_adj = torch.Tensor(test_node_adj[batch_idx])
            batch_node_labels = torch.Tensor(test_node_labels[batch_idx]).long()
            batch_node_targets = torch.Tensor(test_node_targets[batch_idx])
            batch_node_targets_all = torch.Tensor(test_node_targets_all[batch_idx])

            for _ in range(0, max(sequence_length, 1)):
                log_probs, pred, pred_all = model(batch_node_obs, batch_node_adj)

            if with_classification:
                # combine batch and node dimensions for loss calculation
                log_probs_loss = log_probs.reshape(-1, NUM_CLASSES)
                batch_node_labels_loss = batch_node_labels.reshape(-1, 1)

                total_class_loss += cross_entropy(
                    log_probs_loss, batch_node_labels_loss.squeeze(-1)
                ).item()
                total_correct += (
                    (model.get_prediction() == batch_node_labels).sum().item()
                )

            if with_regression:
                pred = pred.reshape(-1, 1)
                batch_node_targets = batch_node_targets.reshape(-1, 1)
                total_reg_loss += mse(pred, batch_node_targets).item()

            if with_regression_all:
                total_reg_all_loss += mse(
                    pred_all[..., : args.num_targets],
                    batch_node_targets_all[..., : args.num_targets],
                ).item()

            pbar.update(next_idx - idx)
            idx = next_idx

    pbar.close()

    if with_classification:
        print(
            f"{total_correct / total_count:.2f} acc, {total_class_loss / total_count} loss"
        )
    if with_regression:
        print(f"Pred loss {total_reg_loss / total_count}")

    if with_regression_all:
        print(f"Pred_all loss {total_reg_all_loss / (total_count * args.num_targets)}")

    return (
        total_correct / total_count,
        total_class_loss / total_count,
        total_reg_loss / total_count,
        total_reg_all_loss / (total_count * args.num_targets),
    )


if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("Training with GPU enabled")

batch_size = 32
clear_cache = args.clear_cache
# whether to cache/save the generated dataset
cache = True

# 10_000 1_000 4_000
num_samples_train = args.num_samples_train
num_samples_test = 1_000
assert num_samples_test <= len(EVAL_SEEDS)
train_iterations = args.iterations
with_classification = False
with_regression = False
with_regression_all = True

sequence_length = args.sequence_length

save_results_filename = args.filename
if save_results_filename is not None and Path(save_results_filename).exists():
    Path(save_results_filename).unlink()

num_nodes = 20
num_packets = 20
if args.num_targets is None:
    args.num_targets = num_nodes

assert 1 <= args.num_targets <= num_nodes

networks_train = Network(
    num_nodes,
    random_topology=True,
    excluded_seeds=EVAL_SEEDS,
)
networks_val = Network(
    num_nodes,
    random_topology=True,
    n_random_seeds=num_samples_test,
    sequential_topology_seeds=True,
    excluded_seeds=EVAL_SEEDS,
)
networks_test = Network(
    num_nodes,
    random_topology=True,
    sequential_topology_seeds=True,
    provided_seeds=EVAL_SEEDS,
)
env = Routing(networks_train, num_packets, EnvironmentVariant.INDEPENDENT)
env_val = Routing(networks_val, num_packets, EnvironmentVariant.INDEPENDENT)
env_test = Routing(networks_test, num_packets, EnvironmentVariant.INDEPENDENT)
env.network.G_weight_key = (
    # number of hops
    # None
    # actual edge lengths (needs more training iterations!)
    "weight"
)
env_val.network.G_weight_key = env.network.G_weight_key
env_test.network.G_weight_key = env.network.G_weight_key
env.reset()
env_val.reset()
env_test.reset()
node_observation_dim = len(env.get_node_observation()[0])

netmon = NetMonSL(node_observation_dim, NUM_CLASSES, num_nodes)
summary_node_obs = torch.tensor(
    env.get_node_observation(), dtype=torch.float32
).unsqueeze(0)
summary_node_adj = torch.tensor(
    env.get_nodes_adjacency(), dtype=torch.float32
).unsqueeze(0)
summary_node_agent = torch.tensor(
    env.get_node_agent_matrix(), dtype=torch.float32
).unsqueeze(0)
print(netmon.netmon.summarize(summary_node_obs, summary_node_adj, summary_node_agent))

print("Loading train dataset..")
train_data = build_or_load_dataset(
    env, num_samples_train, "train.pk", clear_cache, cache
)
print(f"loaded {train_data[0].shape[0]} samples")

print("Loading validation dataset..")
val_data = build_or_load_dataset(
    env_val, num_samples_test, "val.pk", clear_cache, cache
)
print(f"loaded {val_data[0].shape[0]} samples")

print("Loading test dataset..")
test_data = build_or_load_dataset(
    env_test, num_samples_test, "test.pk", clear_cache, cache
)
print(f"loaded {test_data[0].shape[0]} samples")


def validation_callback():
    return test(
        netmon,
        *val_data,
        batch_size,
        with_classification,
        with_regression,
        with_regression_all,
        sequence_length,
        disable_progress_force=True,
    )


# training
total_loss, class_loss, reg_loss, reg_loss_all, validation_results = train(
    netmon,
    *train_data,
    train_iterations,
    batch_size,
    with_classification,
    with_regression,
    with_regression_all,
    sequence_length,
    validation_callback=validation_callback,
)

# save losses
if save_results_filename is not None:
    class_loss = class_loss if with_classification else np.zeros(train_iterations)
    reg_loss = reg_loss if with_regression else np.zeros(train_iterations)
    reg_loss_all = reg_loss_all if with_regression_all else np.zeros(train_iterations)
    df_loss = pd.DataFrame(
        data=list(
            zip(
                np.arange(train_iterations),
                class_loss,
                reg_loss,
                reg_loss_all,
                total_loss,
            )
        ),
        columns=[
            "Iteration",
            "Classification Loss",
            "Regression Loss",
            "Regression Loss All",
            "Total Loss",
        ],
    )
    df_loss.to_hdf(save_results_filename, "loss", mode="a")

    def validation_results_idx(idx):
        return list(map(lambda x: x[idx], validation_results))

    df_validation = pd.DataFrame(
        data=list(
            zip(
                validation_results_idx(0),
                validation_results_idx(1),
                validation_results_idx(2),
                validation_results_idx(3),
                validation_results_idx(4),
            )
        ),
        columns=[
            "Iteration",
            "Accuracy",
            "Classification Loss",
            "Regression Loss",
            "Regression Loss All",
        ],
    )
    df_validation.to_hdf(save_results_filename, "validation", mode="a")


print("Train data eval: ")
train_acc, train_class_loss, train_reg_loss, train_reg_loss_all = test(
    netmon,
    *train_data,
    batch_size,
    with_classification,
    with_regression,
    with_regression_all,
    sequence_length,
)

# testing

print(f"Test data eval: (seq_len={sequence_length})")
test_acc, test_class_loss, test_reg_loss, test_reg_loss_all = test(
    netmon,
    *test_data,
    batch_size,
    with_classification,
    with_regression,
    with_regression_all,
    sequence_length,
)

# for sequence length > 1, also start eval out of train sequence length
# removed this condition
test_sequence_results = []

for seq_len in args.test_sequence_lengths:
    print(f"Extended test data eval (seq_len={seq_len})")
    test_sequence_results.append(
        # append new tuple with sequence length and test results
        (
            seq_len,
            *test(
                netmon,
                *test_data,
                batch_size,
                with_classification,
                with_regression,
                with_regression_all,
                seq_len,
            ),
        )
    )

# save results
if save_results_filename is not None:
    n = len(test_sequence_results)

    def test_sequence_results_idx(idx):
        return list(map(lambda x: x[idx], test_sequence_results))

    df_loss = pd.DataFrame(
        data=list(
            zip(
                ["train"] + ["test"] * (1 + n),
                [train_data[0].shape[0]] + [test_data[0].shape[0]] * (1 + n),
                [sequence_length, sequence_length] + test_sequence_results_idx(0),
                [args.netmon_iterations] * (2 + n),
                [train_acc, test_acc] + test_sequence_results_idx(1),
                [train_class_loss, test_class_loss] + test_sequence_results_idx(2),
                [train_reg_loss, test_reg_loss] + test_sequence_results_idx(3),
                [train_reg_loss_all, test_reg_loss_all] + test_sequence_results_idx(4),
            )
        ),
        columns=[
            "Type",
            "Samples",
            "Sequence Length",
            "Netmon Iterations",
            "Accuracy",
            "Classification Loss",
            "Regression Loss",
            "Regression Loss All",
        ],
    )
    df_loss.to_hdf(save_results_filename, "results", mode="a")
