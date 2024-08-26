import gtsam
import numpy as np
from scipy.spatial.transform import Rotation as R


def estimate_poses_given_rot(
        factors: gtsam.BetweenFactorPose3s,
        rotations: gtsam.Values,
        uncertainty,
        d: int = 3,
):
    """Estimate Poses from measurements, given rotations. From SfmProblem in shonan.
    Arguments:
        factors -- data structure with many BetweenFactorPose3 factors
        rotations {Values} -- Estimated rotations
    Returns:
        Values -- Estimated Poses
    """

    I_d = np.eye(d)

    def R(j):
        return rotations.atRot3(j) if d == 3 else rotations.atRot2(j)

    def pose(R, t):
        return gtsam.Pose3(R, t) if d == 3 else gtsam.Pose2(R, t)

    graph = gtsam.GaussianFactorGraph()
    model = gtsam.noiseModel.Unit.Create(d)

    # Add a factor anchoring t_0
    graph.add(0, I_d, np.zeros((d,)), model)

    # Add a factor saying t_j - t_i = Ri*t_ij for all edges (i,j)
    for idx in range(len(factors)):
        factor = factors[idx]
        keys = factor.keys()
        i, j, Tij = keys[0], keys[1], factor.measured()
        if i == j:
            continue
        model = gtsam.noiseModel.Diagonal.Variances(
            uncertainty[idx] * (1e-2) * np.ones(d)
        )
        measured = R(i).rotate(Tij.translation())
        graph.add(j, I_d, i, -I_d, measured, model)

    # Solve linear system
    translations = graph.optimize()
    # Convert to Values.
    result = gtsam.Values()
    for j in range(rotations.size()):
        tj = translations.at(j)
        result.insert(j, pose(R(j), tj))

    return result


def shonan_averaging(v_num, edges, transformations, uncertainty):
    """
    Input:
    edges: [m, 2], set of directed edges (i, j)
    transformations: [m, 4, 4], T_ij = inv(T_i) @ T_j
        meaning points in S_j transform along T_j to global coord, then inv(T_i) to S_i coord
    """

    edge_num = edges.shape[0]
    factors = []
    # odomModel = gtsam.noiseModel.Diagonal.Variances(np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]))
    new_uncertainty = []
    for i in range(edge_num):
        if edges[i, 1] == edges[i, 0]:
            continue
        odomModel = gtsam.noiseModel.Diagonal.Variances(
            uncertainty[i] * np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])
        )
        factor = gtsam.BetweenFactorPose3(
            edges[i, 0],
            edges[i, 1],
            gtsam.Pose3(transformations[i, :, :]),
            odomModel,
        )
        factors.append(factor)
        new_uncertainty.append(uncertainty[i])
        # print(transformations[i, :, :])
    shonan = gtsam.ShonanAveraging3(gtsam.BetweenFactorPose3s(factors))
    initial = shonan.initializeRandomly()
    try:
        rotations, _ = shonan.run(initial, 3, 10)
        poses = estimate_poses_given_rot(
            factors, rotations, np.array(new_uncertainty), d=3
        )
    except:
        print("shonan did not converge")
        global_pose_results = []
        for i in range(v_num):
            global_pose_result = np.eye(4)
            global_pose_results.append(global_pose_result)
        return np.stack(global_pose_results), 0
    global_pose_results = []
    for i in range(poses.size()):
        global_pose_result = poses.atPose3(i).matrix()
        global_pose_results.append(global_pose_result)
    return np.stack(global_pose_results), 1


if __name__ == "__main__":
    total_num = 3
    global_poses = []
    for i in range(total_num):
        transformation = np.eye(4)
        transformation[:3, :3] = R.random().as_matrix()
        transformation[:3, 3] = np.random.rand(3)
        global_poses.append(transformation)
    global_poses = np.stack(global_poses)
    # global_poses[1, :3, 3] = np.ones(3)
    n = global_poses.shape[0]
    uncertainty = []
    for i in range(n):
        global_poses[n - i - 1, :, :] = (
                np.linalg.inv(global_poses[0, :, :]) @ global_poses[n - i - 1, :, :]
        )
    edges = np.stack(
        [
            np.repeat(np.arange(total_num), total_num),
            np.tile(np.arange(total_num), total_num),
        ]
    ).transpose()
    # edges = np.array([[1, 0], [2, 0]])
    transformations = []
    for i in range(edges.shape[0]):
        transformations.append(
            np.linalg.inv(global_poses[edges[i, 0], :, :])
            @ global_poses[edges[i, 1], :, :]
        )
        # transformations.append(global_poses[edges[i, 1], :, :] @ np.linalg.inv(global_poses[edges[i, 0], :, :]))
        uncertainty.append(1)
    for i in range(total_num):
        edges = np.concatenate([edges, np.array([[total_num, i]])], axis=0)
        transformations.append(np.eye(4))
        uncertainty.append(1e6)
    uncertainty = np.array(uncertainty)
    transformations = np.stack(transformations)
    uncertainty[0] = 0.01
    uncertainty[1] = 0.01
    transformations[0, :, :] = np.array(
        [
            [0.58883767, 0.76988153, -0.24607443, 0.20378149],
            [-0.64633755, 0.26572036, -0.71529048, 0.46243953],
            [-0.48530194, 0.58023712, 0.6540695, 0.18572326],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    transformations[1, :, :] = np.array(
        [
            [0.65344216, -0.75661764, -0.02330413, 0.80860915],
            [-0.74655571, -0.64923189, 0.14543908, 0.18233219],
            [-0.12517156, -0.0776382, -0.98909271, 0.63350485],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    global_pose_results, success = shonan_averaging(
        total_num, edges, transformations, uncertainty
    )
    for i in range(n):
        global_pose_results[n - i - 1, :, :] = (
                np.linalg.inv(global_pose_results[0, :, :])
                @ global_pose_results[n - i - 1, :, :]
        )

    for i in range(n):
        np.set_printoptions(precision=3, suppress=True)
        print(global_poses[i, :, :])
        np.set_printoptions(precision=3, suppress=True)
        print(global_pose_results[i, :, :])
