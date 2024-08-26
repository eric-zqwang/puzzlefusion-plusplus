import numpy as np
from scipy.spatial.transform import Rotation as R

from .pose_graph_utils import connect_graph
from .shonan_averaging import shonan_averaging
from .spanning_tree_alignment import spanning_tree_alignment


def global_alignment(v_num, edges, transformations, uncertainty, method='gtsam'):
    """
        Input:
        edges: [m, 2], set of directed edges (i, j)
        transformations: [m, 4, 4], T_ij = inv(T_i) @ T_j
            meaning points in S_j transform along T_j to global coord, then inv(T_i) to S_i coord
    """
    auxiliary_edges = connect_graph(v_num, edges)
    edges = np.concatenate([edges, auxiliary_edges], axis=0).astype(np.int32)
    auxiliary_transformations = []
    for i in range(auxiliary_edges.shape[0]):
        transformation = np.eye(4)
        transformation[:3, :3] = R.random().as_matrix()
        transformation[:3, 3] = np.random.rand(3)
        auxiliary_transformations.append(transformation)
    auxiliary_transformations = np.stack(auxiliary_transformations)
    # auxiliary_transformations = np.repeat(np.eye(4)[np.newaxis, :, :], auxiliary_edges.shape[0], axis=0)
    transformations = np.concatenate([transformations, auxiliary_transformations], axis=0)
    auxiliary_uncertainty = 1 * np.ones(auxiliary_edges.shape[0])
    uncertainty = np.concatenate([uncertainty, auxiliary_uncertainty])
    # print(edges, transformations, uncertainty)
    if method == 'gtsam':
        global_pose_results, success = shonan_averaging(v_num + 1, edges, transformations, uncertainty)
        if success == 0:
            global_pose_results, success = spanning_tree_alignment(v_num + 1, edges, transformations, uncertainty)
        # print('global_pose_results.shape = ', global_pose_results.shape, v_num)
        for i in range(v_num):
            global_pose_results[v_num - i - 1, :, :] = np.linalg.inv(global_pose_results[0, :, :]) @ \
                                                       global_pose_results[v_num - i - 1, :, :]

        return global_pose_results[:v_num, :, :]

    elif method == 'spanning_tree':
        global_pose_results, success = spanning_tree_alignment(v_num + 1, edges, transformations, uncertainty)
        for i in range(v_num):
            global_pose_results[v_num - i - 1, :, :] = np.linalg.inv(global_pose_results[0, :, :]) @ \
                                                       global_pose_results[v_num - i - 1, :, :]

        return global_pose_results[:v_num, :, :]
    else:
        raise NotImplementedError(f'Global_alignment method {method} not implemented')
