from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError

import itertools
import numpy as np


def locations_to_features_diffs(x_1, y_1, x_2, y_2):
    res = np.array([0.5 + 0.5 * (x_1 - x_2) / 256.0, 0.5 + 0.5 * (y_1 - y_2) / 256.0])
    return res


def build_graphs(P_np: np.ndarray, n: int, n_pad: int = None, edge_pad: int = None):

    A = delaunay_triangulate(P_np[0:n, :])
    edge_num = int(np.sum(A, axis=(0, 1)))

    if n_pad is None:
        n_pad = n
    if edge_pad is None:
        edge_pad = edge_num
    assert n_pad >= n
    assert edge_pad >= edge_num

    edge_list = [[], []]
    features = []
    for i in range(n):
        for j in range(n):
            if A[i, j] == 1:
                edge_list[0].append(i)
                edge_list[1].append(j)
                features.append(locations_to_features_diffs(*P_np[i], *P_np[j]))

    if not features:
        features = np.zeros(shape=(0, 2))

    return np.array(edge_list, dtype=np.int), np.array(features)


def delaunay_triangulate(P: np.ndarray):
    """
    Perform delaunay triangulation on point set P.
    :param P: point set
    :return: adjacency matrix A
    """
    n = P.shape[0]
    if n < 3:
        A = np.ones((n, n)) - np.eye(n)
    else:
        try:
            d = Delaunay(P)
            A = np.zeros((n, n))
            for simplex in d.simplices:
                for pair in itertools.permutations(simplex, 2):
                    A[pair] = 1
        except QhullError as err:
            print("Delaunay triangulation error detected. Return fully-connected graph.")
            print("Traceback:")
            print(err)
            A = np.ones((n, n)) - np.eye(n)
    return A
