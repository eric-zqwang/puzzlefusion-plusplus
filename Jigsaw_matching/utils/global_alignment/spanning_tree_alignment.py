import numpy as np

from .pose_graph_utils import minimum_spanning_tree


def spanning_tree_alignment(v_num, edges, transformations, uncertainty):
    mst_order, mst_pred = minimum_spanning_tree(v_num, edges, uncertainty)
    global_transformation = np.zeros((v_num, 4, 4))
    global_transformation[0, :, :] = np.eye(4)
    hash_map = np.zeros((v_num, v_num, 4, 4))
    for i in range(edges.shape[0]):
        hash_map[edges[i, 0], edges[i, 1], :, :] = transformations[i, :, :]
        hash_map[edges[i, 1], edges[i, 0], :, :] = np.linalg.inv(
            transformations[i, :, :]
        )
    for i in range(1, v_num):
        y = mst_order[i]
        x = mst_pred[y]
        global_transformation[y, :, :] = (
                global_transformation[x, :, :] @ hash_map[x, y, :, :]
        )
    return global_transformation, 1
