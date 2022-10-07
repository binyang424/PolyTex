import numpy as np


def color_cluster(clusters):

    cluster_bounds = clusters["cluster boundary"]
    t_input = clusters["t input"].flatten()
    t_test = clusters["t test"].flatten()

    color = np.zeros_like(clusters["pdf input"])
    for idx, bound in enumerate(cluster_bounds):
        mask1 = t_input >= t_test[bound]
        try:
            mask2 = t_input < t_test[cluster_bounds[idx + 1]]
        except IndexError:
            mask2 = mask1

        mask = mask1 & mask2
        color[mask] = idx

    return color