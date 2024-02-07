import argparse
from collections import defaultdict
import copy
import json
import sys
import traceback
from pathlib import Path
from env.constants import EVAL_SEEDS
from env.environment import reset_and_get_sizes
from env.network import Network

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from eval import evaluate
from replaybuffer import ReplayBuffer
from model import DGN, DQNR, DQN, MLP, CommNet, NetMon
from env.routing import Routing

from env.simple_environment import SimpleEnvironment
from env.wrapper import NetMonWrapper
from policy import EpsilonGreedy
from policy import ShortestPath
from policy import RandomPolicy, SimplePolicy
from buffer import Buffer
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from util import (
    dim_str_to_list,
    filter_dict,
    get_state_dict,
    interpolate_model,
    load_state_dict,
    set_attributes,
    set_seed,
)

parser = argparse.ArgumentParser(
    description="Train and test reinforcement learning agents in graph environments."
)

# environment settings
parser.add_argument(
    "--env-type",
    type=str,
    help="The environment type",
    choices=["routing", "simple"],
    default="routing",
)
parser.add_argument(
    "--n-router",
    type=int,
    help="Number of routers in the routing environment",
    default=20,
)
parser.add_argument(
    "--n-data",
    type=int,
    help="Number of packets in the routing environment",
    default=20,
)
parser.add_argument(
    "--env-var",
    type=int,
    help="Set the environment variant (1: local, 2: k neighbors, 3: global)",
    choices=[1, 2, 3],
    default=1,
)
parser.add_argument(
    "--episode-steps",
    type=int,
    help="Maximum number of steps for an episode",
    default=300,
)
parser.add_argument(
    "--ttl",
    type=int,
    help="Time to live for packets, set to 0 to disable",
    default=0,
)
parser.add_argument(
    "--random-topology",
    type=int,
    help="Use a random topology (1) or default topology (0)",
    # 0: False, 1: True
    choices=[False, True],
    default=True,
)
parser.add_argument(
    "--topology-init-seed",
    type=int,
    help="Init seed for fixed and random topology generation",
    default=476,
)
parser.add_argument(
    "--train-topology-allow-eval-seed",
    help="Allow the use of evaluation seeds during training (e.g. for debugging)",
    dest="train_topology_allow_eval_seed",
    action="store_true",
)
parser.set_defaults(train_topology_allow_eval_seed=False)
parser.add_argument(
    "--num-topologies-train",
    type=int,
    help="Number of random topologies for training (0 for unlimited)",
    default=0,
)
parser.add_argument(
    "--no-congestion",
    help="Disables congestion in the routing environment",
    dest="no_congestion",
    action="store_true",
)
parser.set_defaults(no_congestion=False)
parser.add_argument(
    "--action-mask",
    help="Enables action masking in the routing environment",
    dest="enable_action_mask",
    action="store_true",
)
parser.set_defaults(enable_action_mask=False)

# approach settings
parser.add_argument(
    "--netmon", help="Enables graph observations", dest="netmon", action="store_true"
)
parser.add_argument(
    "--no-netmon",
    help="Disables graph observations (default)",
    dest="netmon",
    action="store_false",
)
parser.set_defaults(netmon=False)
parser.add_argument(
    "--netmon-global",
    help="Enables global pooling of graph observations (only allowed in centralized case)",
    dest="netmon_global",
    action="store_true",
)
parser.set_defaults(netmon_global=False)
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
    "--netmon-startup-iterations",
    type=int,
    help="Number of message passing iterations after environment reset (before first step)",
    default=1,
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

# model settings
parser.add_argument(
    "--model",
    type=str,
    help="Base algorithm/model",
    choices=["dgn", "dqn", "dqnr", "commnet"],
    default="dgn",
)
parser.add_argument(
    "--activation-function",
    type=str,
    help="Activation function used in the model",
    default="leaky_relu",
)
parser.add_argument(
    "--hidden-dim",
    type=str,
    help="Set the encoder dimension(s) that determine the hidden dim. "
    "Examples: '128', '512, 128'.. ",
    default="512,256",
)
parser.add_argument(
    "--num-heads",
    type=int,
    help="Number of agent attention heads (DGN only)",
    default=8,
)
parser.add_argument(
    "--num-attention-layers",
    type=int,
    help="Number of agent attention layers (DGN only)",
    default=2,
)

# training settings
parser.add_argument(
    "--total-steps",
    type=int,
    help="Total number of training steps",
    default=1e6,
)
parser.add_argument(
    "--step-between-train",
    type=int,
    help="Number of steps that are performed between training iterations.",
    default=1,
)
parser.add_argument("--lr", type=float, help="Learning rate", default=1e-4)
parser.add_argument(
    "--step-before-train",
    type=int,
    help="Number of steps that are collected before training",
    default=2000,
)
parser.add_argument("--capacity", type=int, help="Replay memory capacity", default=2e5)
parser.add_argument(
    "--replay-half-precision",
    help="Limit replay memory to half precision",
    dest="replay_half_precision",
    action="store_true",
)
parser.set_defaults(replay_half_precision=False)
parser.add_argument("--gamma", type=float, help="Discount factor", default=0.98)
parser.add_argument(
    "--mini-batch-size", type=int, help="Training mini batch size", default=10
)
parser.add_argument(
    "--epsilon", type=float, help="Initial exploration probability", default=0.6
)
parser.add_argument(
    "--epsilon-decay",
    type=float,
    help="Epsilon decay rate (multiplicative)",
    default=0.996,
)
parser.add_argument(
    "--epsilon-update-freq",
    type=int,
    help="Number of steps between applications of the epsilon decay factor",
    default=100,
)
parser.add_argument(
    "--sequence-length",
    type=int,
    help="Length of sampled sequences during training",
    default=1,
)
parser.add_argument(
    "--att-regularization-coeff",
    type=float,
    help="Attention regularization coefficient (DGN only)",
    default=0.03,
)
parser.add_argument(
    "--aux-loss-coeff",
    type=float,
    help="Auxiliary loss coefficient to enable supervised learning during RL",
    default=0.0,
)
parser.add_argument(
    "--target-update-steps",
    type=int,
    help="Number of steps between target model updates (smooth updates for 0)",
    default=0,
)
parser.add_argument(
    "--tau",
    type=float,
    help="Interpolation factor for smooth target model updates",
    default=0.01,
)
parser.add_argument(
    "--model-checkpoint-steps",
    type=int,
    help="Number of steps between saved model checkpoints",
    default=1e5,
)
parser.add_argument(
    "--comment",
    type=str,
    help="Select a comment that allows to identify the run",
    default="",
)
parser.add_argument(
    "--model-load-path",
    type=str,
    help="Loads a model from the given path",
    default=None,
)
parser.add_argument(
    "--model-load-no-args",
    help="When loading a model, do not automatically overwrite the model's arguments.",
    dest="model_load_no_args",
    action="store_true",
)
parser.set_defaults(model_load_no_args=False)
parser.add_argument(
    "--eval", help="Only run the evaluation", dest="eval", action="store_true"
)
parser.set_defaults(eval=False)
parser.add_argument(
    "--eval-output-dir",
    help="Output eval directory (set to save results, only used with --eval)",
    default=None,
    type=str,
)
parser.add_argument(
    "--eval-output-detailed",
    help="Output more detailed evaluation output for all episodes.",
    dest="eval_output_detailed",
    action="store_true",
)
parser.set_defaults(eval_output_detailed=False)
parser.add_argument(
    "--eval-output-node-state-aux",
    help="Output node state and aux information after eval (WARNING: potentially huge filesize).",
    dest="output_node_state_aux",
    action="store_true",
)
parser.set_defaults(output_node_state_aux=False)
parser.add_argument(
    "--disable-progressbar",
    help="Disables the progress bar",
    dest="disable_progressbar",
    action="store_true",
)
parser.set_defaults(disable_progressbar=False)
parser.add_argument(
    "--eval-episodes", type=int, help="Number of eval episodes", default=1000
)
parser.add_argument(
    "--eval-episode-steps", type=int, help="Maximum steps per eval episode", default=300
)
parser.add_argument(
    "--debug-plots", help="Create debug plots", dest="debug_plots", action="store_true"
)
parser.set_defaults(debug_plots=False)
parser.add_argument(
    "--debug", type=int, help="Debug input to toggle experimental features", default=0
)
parser.add_argument(
    "--device",
    type=str,
    help="Device to use",
    choices=["cpu", "cuda"],
    default="cpu",
)
parser.add_argument("--seed", type=int, help="Seed for the experiment", default=42)
parser.add_argument(
    "--policy",
    type=str,
    help="The policy that should be used, 'heuristic' depends on the given --env-type",
    choices=["heuristic", "random", "trained"],
    default="trained",
)

args = parser.parse_args()
# automatically limit capacity
args.capacity = min(args.total_steps, args.capacity)

# load model arguments automatically
if args.model_load_path and not args.model_load_no_args:
    assert Path(args.model_load_path).exists()
    loaded_dict = torch.load(args.model_load_path, map_location=args.device)
    loaded_model_arg_values = filter_dict(
        loaded_dict["args"],
        [
            "model",
            "hidden_dim",
            "netmon",
            "netmon_dim",
            "netmon_encoder_dim",
            "netmon_iterations",
            "netmon_rnn_type",
            "netmon_agg_type",
            "netmon_global",
            "activation_function",
            "num_heads",
            "num_attention_layers",
        ],
    )
    loaded_model_arg_values.update({"policy": "trained"})
    set_attributes(args, loaded_model_arg_values)

set_seed(args.seed)

# argument checking
if args.sequence_length <= 0:
    raise ValueError(
        f"Invalid sequence length {args.sequence_length}. Must be greater 0."
    )

network = Network(
    n_nodes=args.n_router,
    random_topology=args.random_topology,
    n_random_seeds=args.num_topologies_train,
    topology_init_seed=args.topology_init_seed,
    excluded_seeds=None if args.train_topology_allow_eval_seed else EVAL_SEEDS,
)

# define the environment
if args.env_type == "routing":
    env = Routing(
        network,
        args.n_data,
        args.env_var,
        enable_congestion=not args.no_congestion,
        enable_action_mask=args.enable_action_mask,
        ttl=args.ttl,
    )
elif args.env_type == "simple":
    env = SimpleEnvironment(env_var=args.env_var, random_topology=args.random_topology)
else:
    raise ValueError(f"Unknown environment {args.env_type}")

# get activation function from pytorch
activation_function = getattr(F, args.activation_function)

n_agents, agent_obs_size, n_nodes, node_obs_size = reset_and_get_sizes(env)

# optionally create netmon
if args.netmon:
    netmon = NetMon(
        node_obs_size,
        args.netmon_dim,
        dim_str_to_list(args.netmon_encoder_dim),
        args.netmon_iterations,
        activation_fn=activation_function,
        rnn_type=args.netmon_rnn_type,
        rnn_carryover=args.netmon_rnn_carryover,
        agg_type=args.netmon_agg_type,
        output_neighbor_hidden=True,
        output_global_hidden=args.netmon_global,
    ).to(args.device)
    summary_node_obs = torch.tensor(
        env.get_node_observation(), dtype=torch.float32, device=args.device
    ).unsqueeze(0)
    summary_node_adj = torch.tensor(
        env.get_nodes_adjacency(), dtype=torch.float32, device=args.device
    ).unsqueeze(0)
    summary_node_agent = torch.tensor(
        env.get_node_agent_matrix(), dtype=torch.float32, device=args.device
    ).unsqueeze(0)
    netmon_summary = netmon.summarize(
        summary_node_obs, summary_node_adj, summary_node_agent
    )

    node_state_size = netmon.get_state_size()
    node_aux_size = 0 if env.get_node_aux() is None else len(env.get_node_aux()[0])

    # wrap whole environment with netmon. All agents on this environment will
    # automatically use observations from netmon
    env = NetMonWrapper(env, netmon, args.netmon_startup_iterations)
    _, agent_obs_size, _, _ = reset_and_get_sizes(env)
else:
    netmon = None
    n_nodes = node_obs_size = node_state_size = node_aux_size = 0

hidden_dim = dim_str_to_list(args.hidden_dim)

if args.model == "dgn":
    model = DGN(
        agent_obs_size,
        hidden_dim,
        env.action_space.n,
        args.num_heads,
        args.num_attention_layers,
        activation_function,
    ).to(args.device)
elif args.model == "dqnr":
    model = DQNR(
        agent_obs_size,
        hidden_dim,
        env.action_space.n,
        activation_function,
    )
elif args.model == "commnet":
    model = CommNet(
        agent_obs_size,
        hidden_dim,
        env.action_space.n,
        comm_rounds=2,
        activation_fn=activation_function,
    )
elif args.model == "dqn":
    model = DQN(
        agent_obs_size,
        hidden_dim,
        env.action_space.n,
        activation_function,
    )
else:
    raise ValueError(f"Unknown model type {args.model}")

if args.model_load_path:
    assert Path(args.model_load_path).exists()
    load_state_dict(
        torch.load(args.model_load_path, map_location=args.device),
        model,
        netmon,
    )

model_tar = copy.deepcopy(model).to(args.device)
model = model.to(args.device)
model_has_state = hasattr(model, "state")
aux_model = None

if args.policy == "trained":
    policy = EpsilonGreedy(env, model, env.action_space.n, args)
elif args.policy == "heuristic":
    if args.env_type == "routing":
        policy = ShortestPath(env, None, env.action_space.n, args)
    elif args.env_type == "simple":
        policy = SimplePolicy(env, None, env.action_space.n, args)
    else:
        raise ValueError("Undefined argument")
elif args.policy == "random":
    policy = RandomPolicy(env, None, env.action_space.n, args)
else:
    raise ValueError(f"Unknown policy {args.policy}")

# only evaluate and then exit
if args.eval:
    print(f"Policy: {type(policy).__name__}")
    model.eval()
    if netmon is not None:
        netmon.eval()

    if isinstance(env.get(), Routing) and args.random_topology:
        env.get().network.seeds = EVAL_SEEDS
        env.get().network.sequential_topology_seeds = True

        if args.eval_episodes > len(EVAL_SEEDS):
            print(
                "WARNING: Duplicate eval seeds as number of eval episodes is higher than "
                f"the number of available seeds ({len(EVAL_SEEDS)})!"
            )

    print("Performing Evaluation")
    metrics = evaluate(
        env,
        policy,
        args.eval_episodes,
        args.eval_episode_steps,
        args.disable_progressbar,
        args.eval_output_dir,
        args.eval_output_detailed,
        args.output_node_state_aux,
    )
    print(json.dumps(metrics, indent=4, sort_keys=True, default=str))
    sys.exit(0)

# training
assert (
    args.policy == "trained"
), f"Given policy {args.policy} cannot be used for training."

if netmon is None:
    parameters = list(model.parameters())
else:
    parameters = list(model.parameters()) + list(netmon.parameters())
    if node_aux_size > 0 and args.aux_loss_coeff > 0:
        aux_model = MLP(
            node_state_size,
            (node_state_size, node_aux_size),
            activation_function,
            activation_on_output=False,
        )
        aux_model = aux_model.to(args.device)
        parameters = parameters + list(aux_model.parameters())

optimizer = optim.AdamW(parameters, lr=args.lr)

state_len = model.get_state_len() if model_has_state else 0

buff = ReplayBuffer(
    args.seed,
    int(args.capacity),
    n_agents,
    agent_obs_size,
    state_len,
    n_nodes,
    node_obs_size,
    node_state_size,
    node_aux_size,
    half_precision=args.replay_half_precision,
)

# temporary log variables
log_buffer_size = 1000
log_reward = Buffer(log_buffer_size, (args.n_data,), np.float32)
buffer_plot_last_n = 5000

log_info = defaultdict(lambda: Buffer(log_buffer_size, (1,), np.float32))

next_obs = None
next_adj = None

comment = "_"
if hasattr(env, "env_var"):
    comment += f"R{env.env_var.value}"
else:
    comment += "Simple"

comment += f"_{type(model).__name__}"
if netmon is not None:
    comment += "_netmon"
if args.comment != "":
    comment += f"_{args.comment}"

writer = SummaryWriter(comment=comment)

best_mean_reward = -float("inf")

# torch.autograd.set_detect_anomaly(True)

exception_training = None
exception_evaluation = None

try:
    print("Start training with arguments")
    print(json.dumps(args.__dict__, indent=4, sort_keys=True, default=str))
    print(
        f"Model type: {type(model).__name__}"
        f"{'' if not model_has_state else ' (stateful)'}"
    )
    if netmon is not None:
        print(netmon_summary)
    print(env)
    if model_has_state and args.sequence_length <= 1:
        print("Warning: Training stateful model with sequence length 1.")

    # initialize variables for the transition buffer
    netmon_info = next_netmon_info = (0, 0, 0)
    buffer_state = buffer_node_state = 0
    buffer_node_aux = 0
    # env.render()

    episode_step = None
    current_episode = 0
    episode_done = False
    training_iteration = 0
    for step in tqdm(
        range(1, int(args.total_steps) + 1),
        miniters=100,
        dynamic_ncols=True,
        disable=args.disable_progressbar,
    ):
        model.eval()
        if netmon is not None:
            netmon.eval()

        if episode_step is None or episode_done:
            # reset episode
            episode_step = 0

            obs, adj = env.reset()
            current_episode += 1

            # make sure to reset the states
            last_state = None

        # set the current state
        if model_has_state:
            model.state = last_state

        if netmon is not None:
            buffer_node_state = (
                env.last_netmon_state.cpu().detach().numpy()
                if env.last_netmon_state is not None
                else 0
            )
            netmon_info = env.get_netmon_info()
            if hasattr(env, "get_node_aux"):
                buffer_node_aux = env.get_node_aux()

        # get actions and execute step in environment (note: changes model states)
        joint_actions = policy(obs, adj)

        next_obs, next_adj, reward, done, info = env.step(joint_actions)

        # remember the last state for the buffer and update the state
        if model_has_state:
            buffer_state = last_state.cpu().numpy() if last_state is not None else 0
            done_mask = ~torch.tensor(done, dtype=torch.bool, device=args.device).view(
                1, -1, 1
            )
            last_state = model.state * done_mask

        if netmon is not None:
            next_netmon_info = env.get_netmon_info()

        episode_step += 1
        episode_done = episode_step >= args.episode_steps

        buff.add(
            obs,
            joint_actions,
            reward,
            next_obs,
            adj,
            next_adj,
            done,
            episode_done,
            buffer_state,
            buffer_node_state,
            buffer_node_aux,
            *netmon_info,
            *next_netmon_info,
        )

        obs = next_obs
        adj = next_adj

        # training stats

        # log number of steps for all agents after each episode
        log_reward.insert(reward.mean())
        # get all delays
        if episode_done:
            info = env.get_final_info(info)
        for k, v in info.items():
            log_info[k].insert(v)

        mean_output = {}

        if step % log_buffer_size == 0:
            base_path = Path(writer.get_logdir())
            if args.debug_plots:
                if netmon is not None:
                    buff.save_node_state_diff_plot(
                        base_path / f"z_img_node_diff_{step}.png",
                        buffer_plot_last_n,
                    )
                    buff.save_node_state_std_plot(
                        base_path / f"z_img_node_std_{step}.png", buffer_plot_last_n
                    )
                if (netmon is not None and isinstance(env.env, Routing)) or isinstance(
                    env, Routing
                ):
                    if env.record_distance_map:
                        env.save_distance_map_plot(
                            base_path / f"z_img_spawn_distance_{step}.png"
                        )
                        env.distance_map.clear()
                    if env.random_topology or step == log_buffer_size:
                        import networkx as nx
                        import matplotlib.pyplot as plt

                        nx.draw_networkx(
                            env.G,
                            pos=nx.get_node_attributes(env.G, "pos"),
                            with_labels=True,
                            node_color="pink",
                        )
                        plt.savefig(base_path / f"z_img_topology_{step}.png")
                        plt.clf()

            mean_reward = log_reward.mean()
            log_reward.clear()
            for k, v in log_info.items():
                if log_info[k]._count > 0:
                    mean_output[k] = v.mean()
                    v.clear()

            eps_str = (
                f"  eps: {policy._epsilon:.2f}" if hasattr(policy, "_epsilon") else ""
            )
            tqdm.write(
                f"Episode: {current_episode}"  # print current episode
                f"  step: {step/1000:.0f}k"
                f"  reward: {mean_reward:.2f}"
                f"{''.join(f'  {k}: {v:.2f}' for k, v in mean_output.items())}"
                f"{eps_str}"
                f"{' | BEST' if mean_reward > best_mean_reward else ''}"
            )

            if writer is not None:
                writer.add_scalar("Iteration", training_iteration, step)
                writer.add_scalar("Train/Reward", mean_reward, step)
                for k, v in mean_output.items():
                    writer.add_scalar("Train/" + k.capitalize(), v, step)
                if hasattr(policy, "_epsilon"):
                    writer.add_scalar("Train/Epsilon", policy._epsilon, step)
                writer.add_scalar("Train/Episode", current_episode, step)
                writer.flush()

                if mean_reward > best_mean_reward:
                    torch.save(
                        get_state_dict(model, netmon, args.__dict__),
                        Path(writer.get_logdir()) / "model_best.pt",
                    )
                    best_mean_reward = mean_reward

        if (
            step < args.step_before_train
            or buff.count < args.mini_batch_size
            or step % args.step_between_train != 0
        ):
            continue

        # if (step % 1000) == 0:
        #     eval(env, EpsilonGreedy(env, model, env.actionspace.n, args), args)

        # training
        training_iteration += 1

        loss_q = torch.zeros(1, device=args.device)
        loss_aux = torch.zeros(1, device=args.device)
        loss_att = torch.zeros(1, device=args.device)

        model.train()
        if netmon is not None:
            netmon.train()

        for t, batch in enumerate(
            buff.get_batch(
                args.mini_batch_size,
                device=args.device,
                sequence_length=args.sequence_length,
            )
        ):
            if model_has_state and t == 0:
                # load the state from the beginning
                model.state = batch.agent_state

            if netmon is not None:
                if t == 0:
                    # get state from batch
                    netmon.state = batch.node_state
                else:
                    # reset netmon state if env episode was done in this step
                    netmon.state = last_netmon_state * (  # noqa: F821
                        ~last_batch_episode_done  # noqa: F821
                    ).view(-1, 1, 1)

                    # replace node state with newly calculated netmon state
                    # buff.node_state[batch.idx] = netmon.state.detach().cpu().numpy()

                # run netmon step
                network_obs = netmon(
                    batch.node_obs, batch.node_adj, batch.node_agent_matrix
                )

                # netmon aux loss
                if aux_model is not None:
                    aux_prediction = aux_model(netmon.state)
                    loss_aux = (
                        loss_aux
                        + torch.mean((aux_prediction - batch.node_aux) ** 2)
                        / args.sequence_length
                    )

                # replace observation in place
                net_obs_dim = network_obs.shape[-1]
                batch.obs[:, :, -net_obs_dim:] = network_obs

                # if t > 0:
                #     # update buffer
                #     # buff.next_obs[prev_idx][..., -net_obs_dim:] = (
                #     #     network_obs.detach().cpu().numpy()
                #     # )
                #     buff.next_obs[last_batch_idx] = batch.obs.detach().cpu().numpy()

                # remember true netmon state for gradient calculation
                last_netmon_state = netmon.state
                # done episodes for correct masking
                last_batch_episode_done = batch.episode_done
                last_batch_idx = batch.idx

            q_values = model(batch.obs, batch.adj)

            # run target module
            with torch.no_grad():
                if model_has_state:
                    # hack: set state of target model to next state of current model
                    # with this, we avoid having to store state & next state
                    model_tar.state = model.state.detach()

                if netmon is not None:
                    # note that we use the netmon state from the current model here.
                    # not sure if that is really necessary, we could also save the
                    # next state in the transition tuple and predict current & next
                    # network observations with a single netmon invocation.
                    next_network_obs = netmon(
                        batch.next_node_obs,
                        batch.next_node_adj,
                        batch.next_node_agent_matrix,
                    )
                    next_net_obs_dim = next_network_obs.shape[-1]
                    batch.next_obs[:, :, -next_net_obs_dim:] = next_network_obs

                next_q = model_tar(batch.next_obs, batch.next_adj)
                next_q_max = next_q.max(dim=2)[0]

            if model_has_state:
                # mask agent state when the next step belongs to a new episode
                # -> should happen after the state has been used by target model as
                #    episodes with batch.episode_done have valid next steps and hence
                #    the agent's state should be used in the target model
                state_mask = ~batch.done * (~batch.episode_done).view(-1, 1)
                model.state = model.state * state_mask.unsqueeze(-1)

            # DQN target with 1 step bootstrapping
            # we do not use batch.episode_done here, as the next observation
            # from this transition still belongs to the same episode (i.e. is valid)
            chosen_action_target_q = (
                batch.reward + (~batch.done) * args.gamma * next_q_max
            )

            # original DGN loss on all actions, even unused ones
            q_target = q_values.detach()
            q_target = torch.scatter(
                q_target,
                -1,
                batch.action.unsqueeze(-1),
                chosen_action_target_q.unsqueeze(-1),
            )

            # regular DQN loss just on selected actions
            # q_target = chosen_action_target_q
            # q_values = torch.gather(
            #     q_values, -1, batch.action.unsqueeze(-1).cuda()
            # ).squeeze(-1)

            # combined value loss for each sample in the batch
            td_error = q_values - q_target

            # update q-value loss
            loss_q = loss_q + torch.mean(td_error.pow(2)) / args.sequence_length

            if hasattr(model, "att_weights") and args.att_regularization_coeff > 0:
                # Attention regularization of DGN based on the paper
                # (https://arxiv.org/abs/1810.09202), the official implementation
                # https://github.com/PKU-RL/DGN/blob/72721cb2f0b4b95cf6d17b00758d651b8d7b8f67/Routing/routers_regularization.py#L342
                # and the (as of 26.01.2024 incomplete) torch-based implementation
                # https://github.com/jiechuanjiang/pytorch_DGN/blob/bc728c8f298ff5b1e712baa032a28497b4bd876a/Surviving/DGN%2BR/main.py#L95

                # Note: It looks like the official implementation calculates the loss
                # only for the last attention module. We calculate losses for all given
                # attention weights (can be all of them or only last module, depending
                # on the implementation of the model).

                # first kl_div argument is given in log-probability
                attention = F.log_softmax(torch.stack(model.att_weights), dim=-1)
                # As we run the target model with the next observations, the target
                # model contains the attention weights for the next step.
                target_attention = F.softmax(torch.stack(model_tar.att_weights), dim=-1)
                # remember original shape
                old_att_shape = attention.shape
                # join first dimensions as new batch size for kl div
                attention = attention.view(-1, n_agents)
                target_attention = target_attention.view(-1, n_agents)

                # get pointwise kl divergence
                kl_div = F.kl_div(attention, target_attention, reduction="none")
                # bring tensor back to old shape
                #     (num_layers, batch_size, heads, n_agents_source, n_agents_dest)
                kl_div = kl_div.view(old_att_shape)
                # transpose to
                #     (batch_size, n_agents_source, heads, num_layers, n_agents_dest)
                # and reduce last three dimensions with sum
                kl_div = kl_div.transpose(0, -2).transpose(0, 1).sum(dim=(-1, -2, -3))
                # mask out done (source) agents (because their next step is already
                # in a new episode), reduce loss like "batchmean" and clamp number
                # of not done agents with min 1 to avoid division by zero
                kl_div = (kl_div * ~batch.done).sum() / torch.clamp(
                    (~batch.done).sum(), min=1
                )
                loss_att = loss_att + kl_div / args.sequence_length

        loss = (
            loss_q
            + args.att_regularization_coeff * loss_att
            + args.aux_loss_coeff * loss_aux
        )
        optimizer.zero_grad()
        loss.backward()
        # clip based on https://stackoverflow.com/questions/69427103/gradient-exploding-problem-in-a-graph-neural-network
        torch.nn.utils.clip_grad_value_(parameters, 0.5)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

        if aux_model is not None:
            log_info["loss_aux"].insert(loss_aux.detach().mean().item())

        log_info["q_values"].insert(q_values.detach().mean().item())
        log_info["q_target"].insert(q_target.detach().mean().item())

        log_info["loss"].insert(loss.item())
        # only log q and attention loss if necessary
        if hasattr(model, "att_weights") and args.att_regularization_coeff > 0:
            log_info["loss_q"].insert(loss_q.item())
            log_info["loss_att"].insert(loss_att.item())

        if args.target_update_steps <= 0:
            # smooth target update as in DGN
            interpolate_model(model, model_tar, args.tau, model_tar)
        elif training_iteration % args.target_update_steps == 0:
            # regular target update
            model_tar.load_state_dict(model.state_dict())
            tqdm.write(f"Update network, train iteration {training_iteration}")

        # save checkpoints
        if step % args.model_checkpoint_steps == 0 and writer is not None:
            torch.save(
                get_state_dict(model, netmon, args.__dict__),
                Path(writer.get_logdir()) / f"model_{int(step):_d}.pt",
            )
except Exception as e:
    traceback.print_exc()
    exception_training = e
finally:
    print("Performing clean exit")
    del buff
    if writer is not None:
        try:
            # first reset model & netmon state
            if hasattr(model, "state"):
                model.state = None
            if netmon is not None:
                netmon.state = None

            # evaluate
            if isinstance(env.get(), Routing) and args.random_topology:
                env.get().network.seeds = EVAL_SEEDS
                env.get().network.sequential_topology_seeds = True

                if args.eval_episodes > len(EVAL_SEEDS):
                    print("WARNING: DUPLICATE EVAL SEEDS")

            print("Performing Evaluation")
            metrics = evaluate(
                env,
                policy,
                args.eval_episodes,
                args.eval_episode_steps,
                args.disable_progressbar,
                Path(writer.get_logdir()) / "eval",
                args.eval_output_detailed,
                args.output_node_state_aux,
            )
            print(json.dumps(metrics, indent=4, sort_keys=True, default=str))
            writer.add_hparams(
                hparam_dict=args.__dict__, metric_dict=metrics, run_name="./"
            )
        except Exception as e:
            traceback.print_exc()
            exception_evaluation = e
        finally:
            # save final model
            torch.save(
                get_state_dict(model, netmon, args.__dict__),
                Path(writer.get_logdir()) / "model_last.pt",
            )

            writer.flush()
            writer.close()

# if we encountered any exceptions on the way, raise a runtime error at the end
if exception_training is not None or exception_evaluation is not None:
    if exception_training is not None and exception_evaluation is not None:
        str_ex = "training and evaluation"
    elif exception_training is not None:
        str_ex = "training"
    elif exception_evaluation is not None:
        str_ex = "evaluation"

    raise SystemExit(f"An exception was raised during {str_ex} (see above).")
