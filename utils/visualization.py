import datetime
import time
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from utils import latex_utils as lu
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx

from utils.config import cfg
from utils.decorators import input_to_numpy
from utils.utils import UnNormalize, n_and_l_iter_parallel, lexico_iter


colors = [
    (0.368, 0.507, 0.71),
    (0.881, 0.611, 0.142),
    (0.56, 0.692, 0.195),
    (0.923, 0.386, 0.209),
    (0.528, 0.471, 0.701),
    (0.772, 0.432, 0.102),
    (0.364, 0.619, 0.782),
    (0.572, 0.586, 0.0),
]


def visualize_graph(
    graph, pos, im, suffix, idx, vis_dir, mode="full", edge_colors=None, node_colors=None, true_graph=None
):
    im = np.rollaxis(im, axis=0, start=3)

    network = to_networkx(graph)

    plt.figure()
    plt.imshow(im)

    if mode == "only_edges":
        true_network = to_networkx(true_graph)
        nx.draw_networkx_edges(
            true_network,
            pos=pos,
            arrowstyle="-",
            style="dashed",
            alpha=0.8,
            node_size=15,
            edge_color="white",
            arrowsize=1,
            connectionstyle="arc3,rad=0.2",
        )
        nx.draw_networkx(
            network,
            pos=pos,
            cmap=plt.get_cmap("inferno"),
            node_color="white",
            node_size=15,
            linewidths=1,
            arrowstyle="-",
            edge_color=edge_colors,
            arrowsize=1,
            with_labels=False,
            connectionstyle="arc3,rad=0.2",
            vmin=0.0,
            vmax=1.0,
            width=2.0,
        )
        suffix = suffix + "_" + str(int(time.time() * 100))
    elif mode == "triang":

        node_colors = np.linspace(0, 1, len(network.nodes))
        nx.draw_networkx(
            network,
            pos=pos,
            cmap=plt.get_cmap("Set1"),
            node_color=node_colors,
            node_size=15,
            linewidths=5,
            arrowsize=1,
            with_labels=False,
            vmin=0.0,
            vmax=1.0,
            arrowstyle="-",
        )
    elif mode == "full":

        node_colors = np.linspace(0, 1, len(network.nodes))
        edge_labels = {graph.edge_index[:, i]: f"{i}" for i in range(graph.edge_index.shape[1])}

        nx.draw_networkx(
            network, pos=pos, cmap=plt.get_cmap("Set1"), node_color=node_colors, node_size=100, linewidths=10
        )
        nx.draw_networkx_edge_labels(network, pos=pos, edge_labels=edge_labels, label_pos=0.3)
    elif mode == "only_nodes":
        node_colors = np.linspace(0, 1, len(network.nodes))
        nx.draw_networkx_nodes(
            network, pos=pos, cmap=plt.get_cmap("Set1"), node_color=node_colors, node_size=15, linewidths=1
        )
    elif mode == "nograph":
        pass
    else:
        raise NotImplementedError

    filename = os.path.join(vis_dir, f"{idx}_{suffix}.png")
    plt.savefig(filename)
    plt.close()
    abs_filename = os.path.abspath(filename)
    return abs_filename


@input_to_numpy
def easy_visualize(
    graphs,
    positions,
    n_points,
    images,
    unary_costs,
    quadratic_costs,
    matchings,
    true_matchings,
    string_info,
    reduced_vis,
    produce_pdf=True,
):
    """

    :param graphs: [num_graphs, bs, ...]
    :param positions: [num_graphs, bs, 2, max_n_p]
    :param n_points: [num_graphs, bs, n_p]
    :param images: [num_graphs, bs, size, size]
    :param unary_costs: [num_graphs \choose 2, bs, max_n_p, max_n_p]
    :param quadratic_costs: [num_graphs \choose 2, bs, max_n_p, max_n_p]
    :param matchings: [num_graphs \choose 2, bs, max_n_p, max_n_p]
    """
    positions = [[p[:num] for p, num in zip(pos, n_p)] for pos, n_p in zip(positions, n_points)]
    matchings = [
        [m[:n_p_x, :n_p_y] for m, n_p_x, n_p_y in zip(match, n_p_x_batch, n_p_y_batch)]
        for match, (n_p_x_batch, n_p_y_batch) in zip(matchings, lexico_iter(n_points))
    ]
    true_matchings = [
        [m[:n_p_x, :n_p_y] for m, n_p_x, n_p_y in zip(match, n_p_x_batch, n_p_y_batch)]
        for match, (n_p_x_batch, n_p_y_batch) in zip(true_matchings, lexico_iter(n_points))
    ]

    visualization_string = "visualization"
    latex_file = lu.LatexFile(visualization_string)
    vis_dir = os.path.join(cfg.model_dir, visualization_string)
    unnorm = UnNormalize(cfg.NORM_MEANS, cfg.NORM_STD)
    images = [[unnorm(im) for im in im_b] for im_b in images]

    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    batch = zip(
        zip(*graphs),
        zip(*positions),
        zip(*images),
        zip(*unary_costs),
        zip(*quadratic_costs),
        zip(*matchings),
        zip(*true_matchings),
    )
    for b, (graph_l, pos_l, im_l, unary_costs_l, quadratic_costs_l, matchings_l, true_matchings_l) in enumerate(batch):
        if not reduced_vis:
            files_single = []
            for i, (graph, pos, im) in enumerate(zip(graph_l, pos_l, im_l)):
                f_single = visualize_graph(graph, pos, im, suffix=f"single_{i}", idx=b, vis_dir=vis_dir)
                f_single_simple = visualize_graph(
                    graph, pos, im, suffix=f"single_simple_{i}", idx=b, vis_dir=vis_dir, mode="triang"
                )
                files_single.append(f_single)
                files_single.append(f_single_simple)
            latex_file.add_section_from_figures(
                name=f"Single Graphs ({b})", list_of_filenames=files_single, common_scale=0.7
            )

        files_mge = []
        for (
            unary_c,
            quadratic_c,
            matching,
            true_matching,
            (graph_src, graph_tgt),
            (pos_src, pos_tgt),
            (im_src, im_tgt),
            (i, j),
        ) in n_and_l_iter_parallel(
            n=[unary_costs_l, quadratic_costs_l, matchings_l, true_matchings_l], l=[graph_l, pos_l, im_l], enum=True
        ):
            im_mge, p_mge, graph_mge, edges_corrct_mge, node_colors_mge, true_graph = merge_images_and_graphs(
                graph_src, graph_tgt, pos_src, pos_tgt, im_src, im_tgt, new_edges=matching, true_edges=true_matching
            )
            f_mge = visualize_graph(
                graph_mge,
                p_mge,
                im_mge,
                suffix=f"mge_{i}-{j}",
                idx=b,
                vis_dir=vis_dir,
                mode="only_edges",
                edge_colors=[colors[2] if corr else colors[3] for corr in edges_corrct_mge],
                node_colors=node_colors_mge,
                true_graph=true_graph,
            )
            files_mge.append(f_mge)

            if not reduced_vis:
                f_mge_nodes = visualize_graph(
                    graph_mge,
                    p_mge,
                    im_mge,
                    suffix=f"mge_nodes_{i}-{j}",
                    idx=b,
                    vis_dir=vis_dir,
                    mode="only_nodes",
                    edge_colors=[colors[2] if corr else colors[3] for corr in edges_corrct_mge],
                    node_colors=node_colors_mge,
                    true_graph=true_graph,
                )
                files_mge.append(f_mge_nodes)
                costs_and_matchings = dict(
                    unary_cost=unary_c, quadratic_cost=quadratic_c, matchings=matching, true_matching=true_matching
                )
                for key, value in costs_and_matchings.items():
                    latex_file.add_section_from_dataframe(
                        name=f"{key} ({b}, {i}-{j})", dataframe=pd.DataFrame(value).round(2)
                    )

        latex_file.add_section_from_figures(name=f"Matched Graphs ({b})", list_of_filenames=files_mge, common_scale=0.7)

    time = "{date:%Y-%m-%d_%H-%M-%S}".format(date=datetime.datetime.now())
    suffix = f"{string_info}_{time}"
    output_file = os.path.join(vis_dir, f"{visualization_string}_{suffix}.pdf")
    if produce_pdf:
        latex_file.produce_pdf(output_file=output_file)


def merge_images_and_graphs(graph_src, graph_tgt, p_src, p_tgt, im_src, im_tgt, new_edges, true_edges):
    pos_offset = (im_src.shape[1], 0)
    merged_pos = np.concatenate([p_src, p_tgt + np.array([pos_offset] * p_tgt.shape[0])])
    merged_im = np.concatenate([im_src, im_tgt], 2)
    merged_graph, edges_correct, node_colors, true_graph = merge_graphs(graph_src, graph_tgt, new_edges, true_edges)
    return merged_im, merged_pos, merged_graph, edges_correct, node_colors, true_graph


def merge_graphs(graph1, graph2, new_edges, true_edges):
    merged_x = torch.cat([graph1.x, graph2.x], 0)

    def color_gen():
        for i in np.linspace(0.4, 1, max(new_edges.shape)):
            yield i

    edge_list = [[], []]
    true_edge_list = [[], []]
    edges_correct = []
    node_colors = np.zeros(merged_x.shape[0])
    offset = new_edges.shape[0]
    color = color_gen()
    for i in range(new_edges.shape[0]):
        for j in range(new_edges.shape[1]):
            if new_edges[i, j] == 1:
                edge_list[0].append(i)
                edge_list[1].append(j + offset)
                edges_correct.append(true_edges[i, j])
            if true_edges[i, j]:
                c = next(color)
                node_colors[i], node_colors[j + offset] = c, c
                true_edge_list[0].append(i)
                true_edge_list[1].append(j + offset)

    new_edges = torch.tensor(edge_list, device=graph1.edge_index.device)
    true_edges = torch.tensor(true_edge_list, device=graph1.edge_index.device)
    merged_graph = Data(x=merged_x, edge_index=new_edges)
    true_merged_graph = Data(x=merged_x, edge_index=true_edges)
    return merged_graph, np.array(edges_correct), node_colors, true_merged_graph
