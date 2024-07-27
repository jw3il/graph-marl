from collections import defaultdict
import json
import lzma
from pathlib import Path
import pickle
from typing import Any, Dict, List, NamedTuple, Optional, Union
from matplotlib import pyplot as plt
import networkx as nx

import numpy as np
from tqdm import tqdm

from env.routing import Routing


class StepStats(NamedTuple):
    n: int
    obs: np.ndarray
    adj: np.ndarray
    act: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    info: dict
    node_state: Optional[np.ndarray]
    node_aux: Optional[np.ndarray]


class EpisodeStats(NamedTuple):
    steps: List[StepStats]
    aux: Dict[str, Any]


def evaluate(
    env,
    policy,
    episodes,
    steps_per_episode,
    disable_progressbar=False,
    output_dir: Optional[Union[Path, str]] = None,
    output_detailed=False,
    output_node_state_aux=False,
):
    if output_dir is not None:
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    episode_stats = []

    if hasattr(policy, "eval"):
        policy.eval()

    if hasattr(env, "set_eval_info"):
        env.set_eval_info(True)

    # perform evaluation
    # print("Performing Evaluation")
    for ep in tqdm(range(episodes), disable=disable_progressbar):
        step_stats = []
        aux_stats = {}
        obs, adj = env.reset()

        # print(f"Episode {ep} seed was {env.current_topology_seed}")

        # reset all agents
        if hasattr(policy, "reset"):
            policy.reset(1)

        if hasattr(policy, "reset_episode"):
            policy.reset_episode()

        for step in range(steps_per_episode):
            if hasattr(env, "netmon"):
                node_state = env.netmon.state.detach().cpu().squeeze(0).numpy()
                if node_state.size == 0:
                    node_state = np.zeros((obs.shape[0], 1))
            else:
                node_state = np.zeros((obs.shape[0], 1))

            if output_node_state_aux:
                node_aux = env.get_node_aux()
            else:
                node_aux = None

            actions = policy(obs, adj)
            next_obs, next_adj, reward, done, info = env.step(actions)

            if step + 1 == steps_per_episode:
                # also add delays of agents that did not arrive
                info = env.get_final_info(info)

            # manual eval experiments

            # experiment from the paper with selected bottleneck link
            # if step == 50:
            #     print(env.network.randomize_edge_weights("bottleneck-971182936"))

            # stop adapting netmon after some steps
            # if step == 25:
            #     env.freeze()

            # if (step + 1) % 100 == 0:
            #    env.current_netmon_state = None

            step_stats.append(
                StepStats(
                    step, obs, adj, actions, reward, done, info, node_state, node_aux
                )
            )

            # reset done agents
            if hasattr(policy, "reset"):
                policy.reset(done)

            obs = next_obs
            adj = next_adj

        if isinstance(env.get(), Routing):
            aux_stats["distance_map"] = env.distance_map.copy()
            aux_stats["sum_packets_per_node"] = env.sum_packets_per_node
            aux_stats["sum_packets_per_edge"] = env.sum_packets_per_edge
            aux_stats["G"] = env.network.G.copy()
            env.distance_map.clear()

        episode_stats.append(EpisodeStats(step_stats, aux_stats))

    if hasattr(env, "set_eval_info"):
        env.set_eval_info(False)

    eval_metrics = get_eval_metrics(episode_stats)

    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        with open(output_dir / "metrics.json", "w+") as f:
            json.dump(eval_metrics, f, indent=4, sort_keys=True, default=str)

        if isinstance(env.get(), Routing):
            create_routing_plots(
                episode_stats, output_dir, output_detailed, output_node_state_aux
            )

    return eval_metrics


def get_eval_metrics(episode_stats: List[EpisodeStats]):
    stats_lists = defaultdict(list)
    # join stats for each step in each episode
    for episode in episode_stats:
        for step in episode.steps:
            for k, v in step.info.items():
                if isinstance(v, list):
                    stats_lists[k] += v
                else:
                    stats_lists[k].append(v)

            for r in step.reward:
                stats_lists["reward"].append(r)

    # calculate mean
    metrics = dict()
    for k, v in stats_lists.items():
        if len(stats_lists[k]) > 0:
            v_arr = np.array(v)
            metrics[k + "_mean"] = v_arr.mean()
        else:
            metrics[k + "_mean"] = float("inf")

    return metrics


def save_distance_map_plot(distance_map, filename):
    if len(distance_map) == 0:
        return

    X = np.sort(list(distance_map.keys()))
    Y = np.zeros_like(X, dtype=float)
    Y_err = np.zeros_like(X)
    for i, x in enumerate(X):
        Y_arr = np.array(distance_map[x])
        Y[i] = Y_arr.mean()
        Y_err[i] = Y_arr.std()

    plt.clf()
    plt.plot(X, Y, label="Agent")
    plt.plot(X, X, label="Lower Bound")
    plt.xlabel("Shortest path [steps]")
    plt.ylabel("Agent path [steps]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")


def save_packet_location_graph(
    G,
    sum_packets_per_node,
    sum_packets_per_edge,
    num_steps,
    filename,
):
    plt.clf()
    pos = nx.drawing.spring_layout(G, seed=1337)
    # pos = nx.get_node_attributes(G, "pos")
    edge_weight = np.array([data["weight"] for n1, n2, data in G.edges(data=True)])
    nx_edges = nx.draw_networkx_edges(
        G,
        pos=pos,
        width=4,
        edge_color=sum_packets_per_edge / (np.sum(sum_packets_per_edge) * edge_weight),
        edge_cmap=plt.get_cmap("viridis"),
    )
    plt.colorbar(nx_edges, label="Normalized edge utilization")
    nx_nodes = nx.draw_networkx_nodes(
        G,
        pos=pos,
        node_color=sum_packets_per_node / np.sum(sum_packets_per_node),
        cmap=plt.get_cmap("viridis"),
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels=dict([(i, i) for i in range(G.order())]),
    )
    plt.colorbar(nx_nodes, label="Normalized node utilization")
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=nx.get_edge_attributes(G, "weight"),
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.0, edgecolor="white"),
    )
    # remove border around network
    plt.gca().axis("off")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")


def ddlist():
    """
    Default dict of lists, required for pickle.

    :return: a defaultdict of lists.
    """
    return defaultdict(list)


def create_routing_plots(
    episode_stats: List[EpisodeStats],
    output_dir: Path,
    output_detailed,
    output_node_state_aux,
):
    d = defaultdict(ddlist)

    # join stats for each step in each episode over all nodes
    for episode_i, episode in enumerate(episode_stats):
        last_node_state = 0
        last_step = None

        if output_detailed:
            save_distance_map_plot(
                episode.aux["distance_map"],
                output_dir / f"ep_{episode_i}_distance_map.png",
            )
            save_packet_location_graph(
                episode.aux["G"],
                episode.aux["sum_packets_per_node"],
                episode.aux["sum_packets_per_edge"],
                len(episode.steps),
                output_dir / f"ep_{episode_i}_packet_heatmap.png",
            )

        d["G"][episode_i] = episode.aux["G"]
        d["sum_packets_per_node"][episode_i] = episode.aux["sum_packets_per_node"]
        d["sum_packets_per_edge"][episode_i] = episode.aux["sum_packets_per_edge"]

        for k, v in episode.aux["distance_map"].items():
            d["distance_map"][k] += v

        if output_detailed:
            d["episode_throughput"][episode_i] = defaultdict(list)
        for step in episode.steps:
            d["episode_done_ids"][episode_i] += step.done.nonzero()[0].tolist()
            if output_detailed:
                # episode-wise stats
                d["episode_throughput"][episode_i][step.n].append(
                    step.info["throughput"]
                )

            # step wise stats
            d["feature_diffs"][step.n].append(
                np.mean(np.abs(step.node_state - last_node_state), axis=0)
            )
            d["feature_diffs_mean"][step.n].append(
                np.mean(np.abs(step.node_state - last_node_state))
            )
            d["feature_diffs_max"][step.n].append(
                np.max(np.abs(step.node_state - last_node_state))
            )
            d["reward"][step.n].append(np.mean(step.reward))
            d["feature_std"][step.n].append(np.std(step.node_state, axis=0))
            d["feature_mean"][step.n].append(np.mean(step.node_state, axis=0))
            d["dropped"][step.n].append(step.info["dropped"])
            d["throughput"][step.n].append(step.info["throughput"])
            d["blocked"][step.n].append(step.info["blocked"])
            d["total_edge_load"][step.n].append(step.info["total_edge_load"])

            if len(step.info["delays_arrived"]) > 0:
                d["delays_arrived_mean"][step.n].append(
                    np.mean(step.info["delays_arrived"])
                )

            if len(step.info["spr"]) > 0:
                d["spr_mean"][step.n].append(np.mean(step.info["spr"]))
                d["spr_min"][step.n].append(np.min(step.info["spr"]))

            d["looped"][step.n].append(step.info["looped"])

            if last_step is not None:
                for a in range(step.done.shape[0]):
                    if not step.done[a]:
                        diff = np.abs(
                            step.info["packet_distances"][a]
                            - last_step.info["packet_distances"][a]
                        )
                        d["packet_distance_delta"][step.n].append(diff)

            d["occupied_edges"][step.n].append(step.info["occupied_edges"])
            d["packets_on_edges"][step.n].append(step.info["packets_on_edges"])
            d["total_packet_size"][step.n].append(step.info["total_packet_size"])
            d["packet_distance_mean"][step.n].append(
                np.mean(step.info["packet_distances"])
            )
            d["packet_distance_max"][step.n].append(
                np.max(step.info["packet_distances"])
            )
            for a in range(step.act.shape[0]):
                packet_size = step.info["packet_sizes"][a]
                action = step.act[a]
                d["action_to_packet_size"][action].append(packet_size)

            last_node_state = step.node_state
            last_step = step

    # aggregate done stats
    n_agents = episode_stats[0].steps[0].act.shape[0]
    episode_done_agents = np.zeros((len(episode_stats), n_agents))
    for i in range(len(episode_stats)):
        done_agents = list(np.unique(d["episode_done_ids"][i]))
        episode_done_agents[i, done_agents] = 1

    d["episode_done_agents"] = episode_done_agents

    def plot_attribute(step_value_dict, ylabel, filename, start=0, end=None):
        # automatically try to resolve dict if it is a key
        if isinstance(step_value_dict, str):
            step_value_dict = d[step_value_dict]

        # plot diffs
        x = list(step_value_dict.keys())
        x = np.array(sorted(x))
        mean = np.zeros(len(x))
        std = np.zeros(len(x))

        for i, (step_i, val) in enumerate(step_value_dict.items()):
            val_np = np.array(val)
            mean[i] = val_np.mean()
            std[i] = val_np.std()

        plt.fill_between(
            x[start:end], (mean - std)[start:end], (mean + std)[start:end], alpha=0.2
        )
        plt.plot(x[start:end], mean[start:end])
        plt.xlabel("Steps")
        plt.ylabel(ylabel)
        plt.savefig(filename, bbox_inches="tight")
        plt.clf()

    plot_attribute("reward", "Mean reward", output_dir / "reward.png")
    plot_attribute(
        "feature_diffs",
        "Mean node state difference",
        output_dir / "node_diff.png",
        start=1,
    )
    plot_attribute(
        "feature_diffs_mean",
        "Mean node state difference",
        output_dir / "node_diff_mean.png",
        start=1,
    )
    plot_attribute(
        "feature_diffs_max",
        "Mean node state difference",
        output_dir / "node_diff_max.png",
        start=1,
    )
    plot_attribute("throughput", "Throughput", output_dir / "throughput.png")
    plot_attribute("blocked", "Mean blocked", output_dir / "blocked.png")
    plot_attribute("total_edge_load", "Total edge load", output_dir / "edge_load.png")
    plot_attribute(
        "occupied_edges", "Occupied edges", output_dir / "occupied_edges.png"
    )
    plot_attribute(
        "packets_on_edges", "Packets on edges", output_dir / "packets_on_edges.png"
    )
    plot_attribute(
        "total_packet_size", "Total packet size", output_dir / "packet_size.png"
    )
    plot_attribute(
        "delays_arrived_mean",
        "Delays of arrived packets",
        output_dir / "delays_arrived.png",
    )
    plot_attribute("spr_mean", "Mean spr", output_dir / "spr_mean.png")
    plot_attribute("spr_min", "Min spr", output_dir / "spr_min.png")
    plot_attribute(
        "packet_distance_mean", "Mean distance", output_dir / "distance_mean.png"
    )
    plot_attribute(
        "packet_distance_max", "Max distance", output_dir / "distance_max.png"
    )
    plot_attribute(
        "packet_distance_delta", "Distance delta", output_dir / "distance_delta.png"
    )
    plot_attribute("looped", "Looped", output_dir / "looped.png")
    plot_attribute("dropped", "Dropped", output_dir / "dropped.png")

    # plot packet size to action..plot
    for action in d["action_to_packet_size"]:
        plt.hist(d["action_to_packet_size"][action], bins=100)
        plt.xlabel(f"Packet sizes for action {action}")
        plt.ylabel("Counts")
        plt.savefig(output_dir / f"packet_size_act_{action}.png", bbox_inches="tight")
        plt.clf()

    def plot_done_hists(x, episode_id):
        plt.hist(x, bins=20)
        plt.xlabel("Packet id")
        plt.ylabel("Done count")
        plt.savefig(output_dir / f"ep_{episode_id}_done_hist.png", bbox_inches="tight")
        plt.clf()

    if output_detailed:
        for i in range(len(episode_stats)):
            plot_done_hists(d["episode_done_ids"][i], i)
            plot_attribute(
                d["episode_throughput"][i],
                "Throughput",
                output_dir / f"ep_{i}_throughput.png",
            )

    save_distance_map_plot(d["distance_map"], output_dir / "distance_map.png")

    # save selected stats for combined plotting (paper)
    with lzma.open(output_dir / "lzma_d.pk", "wb") as f:
        pickle.dump(
            {
                k: d[k]
                for k in [
                    "feature_diffs_mean",
                    "feature_diffs_max",
                    "throughput",
                    "delays_arrived_mean",
                    "looped",
                    "reward",
                    "dropped",
                    "episode_done_agents",
                    "G",
                    "sum_packets_per_node",
                    "sum_packets_per_edge",
                ]
            },
            f,
        )

    if output_node_state_aux:
        all_node_states = np.stack(
            [
                np.stack([step.node_state for step in episode.steps])
                for episode in episode_stats
            ]
        )
        all_node_aux = np.stack(
            [
                np.stack([step.node_aux for step in episode.steps])
                for episode in episode_stats
            ]
        )

        np.savez_compressed(
            output_dir / "node_state_aux",
            node_state=all_node_states,
            node_aux=all_node_aux,
        )

    # plot node state content
    # max_episode_steps = len(episode_step_stats[0])
    # node_std_img = np.zeros((max_episode_steps, len(feature_std[0][0])))
    # node_mean_img = np.zeros((max_episode_steps, len(feature_std[0][0])))

    # for step in feature_std.keys():
    #     node_std_img[step] = np.array(feature_std[step]).mean(axis=0)
    #     node_mean_img[step] = np.array(feature_mean[step]).mean(axis=0)

    # fig, axs = plt.subplots(2, 1, sharex=True)
    # axs[0].imshow(node_std_img)
    # axs[1].imshow(node_mean_img)
    # plt.show()
