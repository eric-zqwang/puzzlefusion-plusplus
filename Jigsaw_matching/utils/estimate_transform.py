import numpy as np
import open3d as o3d
import torch

from .pc_utils import to_o3d_pcd, to_o3d_feats, to_tensor, mutual_selection


def run_ransac(xyz_i, xyz_j, corr_idx):
    """
    Ransac based estimation of the transformation paramaters of the congurency transformation. Estimates the
    transformation parameters thtat map xyz0 to xyz1. Implementation is based on the open3d library
    (http://www.open3d.org/docs/release/python_api/open3d.registration.registration_ransac_based_on_correspondence.html)

    Args:
    xyz_i (numpy array): coordinates of the correspondences from the first point cloud [n,3]
    xyz_j (numpy array): coordinates of the correspondences from the second point cloud [n,3]
    Returns:
    trans_param (float): mean registration precision over the scene (not so important because it can be increased see papers)
    recall (float): mean registration recall over the scene (deciding parameter for the performance of the algorithm)
    """

    # Distance threshold as specificed by 3DMatch dataset
    distance_threshold = 0.05

    # Convert the point to an open3d PointCloud object
    xyz0 = o3d.geometry.PointCloud()
    xyz1 = o3d.geometry.PointCloud()

    xyz0.points = o3d.utility.Vector3dVector(xyz_i)
    xyz1.points = o3d.utility.Vector3dVector(xyz_j)

    # Correspondences are already sorted
    # corr_idx = np.tile(np.expand_dims(np.arange(len(xyz0.points)), 1), (1, 2))
    corrs = o3d.utility.Vector2iVector(corr_idx)

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source=xyz0,
        target=xyz1,
        corres=corrs,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=3,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            50000, 2500
        ),
    )

    trans_mat = result_ransac.transformation
    return trans_mat


def get_corr_from_mat(mat):
    if isinstance(mat, np.ndarray):
        corr = np.vstack(mat.nonzero()).transpose(1, 0)
    else:
        assert isinstance(mat, torch.Tensor)
        corr = np.vstack(mat.nonzero())
    return corr


def get_trans_from_mat(pc_src, pc_tgt, mat):
    corr = get_corr_from_mat(mat) # correspondance matrix
    trans_mat = run_ransac(pc_src, pc_tgt, corr)
    return trans_mat, corr


def get_trans_from_corr(pc_src, pc_tgt, corr):
    """
    pc_src: [N1, 3], positions of source point cloud
    pc_tgt: [N2, 3], positions of target point cloud
    corr: [N', 2], each row [idx1, idx2] matches point idx1 in source to point idx2 in target
    """
    trans_mat = run_ransac(pc_src, pc_tgt, corr)
    return trans_mat


def ransac_pose_estimation(
        src_pcd,
        tgt_pcd,
        src_feat,
        tgt_feat,
        mutual=False,
        distance_threshold=0.05,
        ransac_n=3,
):
    """
    RANSAC pose estimation with two checkers
    We follow D3Feat to set ransac_n = 3 for 3DMatch and ransac_n = 4 for KITTI.
    For 3DMatch dataset, we observe significant improvement after changing ransac_n from 4 to 3.
    """
    if mutual:
        if torch.cuda.device_count() >= 1:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        src_feat, tgt_feat = to_tensor(src_feat), to_tensor(tgt_feat)
        scores = torch.matmul(
            src_feat.to(device), tgt_feat.transpose(0, 1).to(device)
        ).cpu()
        selection = mutual_selection(scores[None, :, :])[0]
        row_sel, col_sel = np.where(selection)
        corrs = o3d.utility.Vector2iVector(np.array([row_sel, col_sel]).T)
        src_pcd = to_o3d_pcd(src_pcd)
        tgt_pcd = to_o3d_pcd(tgt_pcd)
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            source=src_pcd,
            target=tgt_pcd,
            corres=corrs,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False
            ),
            ransac_n=4,
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                50000, 1000
            ),
        )
    else:
        src_pcd = to_o3d_pcd(src_pcd)
        tgt_pcd = to_o3d_pcd(tgt_pcd)
        src_feats = to_o3d_feats(src_feat)
        tgt_feats = to_o3d_feats(tgt_feat)

        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_pcd,
            tgt_pcd,
            src_feats,
            tgt_feats,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False
            ),
            ransac_n,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9
                ),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold
                ),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000),
        )

    return result_ransac.transformation


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation as R

    n = 400
    pc1 = np.random.random((n, 3))
    pc2 = pc1.copy()
    trans_mat = np.zeros((4, 4))
    trans_mat[:3, :3] = R.random().as_matrix()
    trans_mat[:3, 3] = np.random.random(3)
    trans_mat[3, 3] = 1
    pc1 = np.concatenate([pc1, np.ones((pc1.shape[0], 1))], axis=1)
    pc1 = trans_mat @ pc1.transpose(1, 0)
    pc1 = pc1[:3, :].transpose(1, 0)
    corr = np.vstack([np.arange(n), np.arange(n)]).transpose(1, 0)
    trans_mat2 = get_trans_from_corr(pc1, pc2, corr)
    print(trans_mat)
    print(trans_mat2)
    print(np.linalg.inv(trans_mat2))
