import copy

import torch

from .chamfer import chamfer_distance
from .loss import _valid_mean
from .transforms import transform_pc
import pytorch3d.transforms as transforms


# @torch.no_grad()
# def calc_part_acc(pts, trans1, trans2, rot1, rot2, valids, ret_cd=False):
#     """Compute the `Part Accuracy` in the paper.

#     We compute the per-part chamfer distance, and the distance lower than a
#         threshold will be considered as correct.

#     Args:
#         pts: [B, P, N, 3], model input point cloud to be transformed
#         trans1: [B, P, 3]
#         trans2: [B, P, 3]
#         rot1: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
#         rot2: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
#         valids: [B, P], 1 for input parts, 0 for padded parts
#         ret_cd: whether return chamfer distance

#     Returns:
#         [B], accuracy per data in the batch
#     """
#     B, P = pts.shape[:2]

#     pts1 = transform_pc(trans1, rot1, pts)  # [B, P, N, 3]
#     pts2 = transform_pc(trans2, rot2, pts)

#     pts1 = pts1.flatten(0, 1)  # [B*P, N, 3]
#     pts2 = pts2.flatten(0, 1)
#     dist1, dist2 = chamfer_distance(pts1, pts2)  # [B*P, N]
#     loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
#     loss_per_data = loss_per_data.view(B, P).type_as(pts)

#     # part with CD < `thre` is considered correct
#     thre = 0.01
#     acc = (loss_per_data < thre) & (valids == 1)
#     # the official code is doing avg per-shape acc (not per-part)
#     acc = acc.sum(-1) / (valids == 1).sum(-1)
#     if ret_cd:
#         cd = loss_per_data.sum(-1) / (valids == 1).sum(-1)
#         return acc, cd
#     return acc

from chamferdist import ChamferDistance
@torch.no_grad()
def calc_part_acc(pts, trans1, trans2, rot1, rot2, valids, ret_cd=False):
    """Compute the `Part Accuracy` in the paper.

    We compute the per-part chamfer distance, and the distance lower than a
        threshold will be considered as correct.

    Args:
        pts: [B, P, N, 3], model input point cloud to be transformed
        trans1: [B, P, 3], pred_translation
        trans2: [B, P, 3], gt_translation
        rot1: [B, P, 4], Rotation3D, quat or rmat
        rot2: [B, P, 4], Rotation3D, quat or rmat
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], accuracy per data in the batch
    """
    chamfer_distance = ChamferDistance()
    B, P = pts.shape[:2]

    pts1 = transform_pc(trans1, rot1, pts)  # [B, P, N, 3]
    pts2 = transform_pc(trans2, rot2, pts)

    pts1 = pts1.flatten(0, 1)  # [B*P, N, 3]
    pts2 = pts2.flatten(0, 1)
    loss_per_data = chamfer_distance(pts1, pts2, bidirectional=True, 
                                    point_reduction="mean", batch_reduction=None,)  # [B*P, N]
    loss_per_data = loss_per_data.view(B, P).type_as(pts)

    # part with CD < `thre` is considered correct
    thre = 0.01
    acc_per_part = (loss_per_data < thre) & (valids == 1)
    # the official code is doing avg per-shape acc (not per-part)
    acc = acc_per_part.sum(-1) / (valids == 1).sum(-1)
    if ret_cd:
        cd = loss_per_data.sum(-1) / (valids == 1).sum(-1)
        return acc, cd
    return acc

# @torch.no_grad()
# def calc_shape_cd(pts, trans1, trans2, rot1, rot2, valids):
#     chamfer_distance = ChamferDistance()
#     B, P, N, _ = pts.shape
    
#     valid_mask = valids[..., None, None]  # [B, P, 1, 1]
    
#     pts = pts.detach().clone()
    
#     pts = pts.masked_fill(valid_mask == 0, 1e3)
    
#     pts1 = transform_pc(trans1, rot1, pts)  # [B, P, N, 3]
#     pts2 = transform_pc(trans2, rot2, pts)
    
#     shape1 = pts1.flatten(1, 2)
#     shape2 = pts2.flatten(1, 2)
    
#     shape_cd = chamfer_distance(
#         shape1, 
#         shape2, 
#         bidirectional=True, 
#         point_reduction=None, 
#         batch_reduction=None
#     )
    
#     shape_cd = shape_cd.view(B, P, N).mean(-1)
#     shape_cd = _valid_mean(shape_cd, valids)
    
#     return shape_cd


@torch.no_grad()
def calc_shape_cd(pts, n_pcs, trans1, rot1, gt_pcs, valids):
    chamfer_distance = ChamferDistance()
    num_parts = valids.sum(-1).to(torch.int32)
    
    shape_cd = []
    for b in range(pts.shape[0]):
        index = 0
        final_pts = []
        for i in range(num_parts[b].item()):
            c_n_pcs = n_pcs[b, i]
            c_pts = pts[b, index:index+c_n_pcs]
            c_trans = trans1[b, i]
            c_rots = rot1[b, i].to_quat()

            c_pts = transforms.quaternion_apply(c_rots, c_pts)
            c_pts = c_pts + c_trans
            final_pts.append(c_pts)
            index += n_pcs[0][i]
        final_pts = torch.cat(final_pts, dim=0)
        gt_pc = gt_pcs[b]
        cd = chamfer_distance(final_pts.unsqueeze(0), gt_pc.unsqueeze(0), bidirectional=True, 
                                    point_reduction="mean", batch_reduction=None,)
        shape_cd.append(cd)

    shape_cd = torch.stack(shape_cd, dim=0).squeeze(1)

    
    return shape_cd


@torch.no_grad()
def calc_connectivity_acc(trans, rot, contact_points):
    """Compute the `Connectivity Accuracy` in the paper.

    We transform pre-computed connected point pairs using predicted pose, then
        we compare the distance between them.
    Distance lower than a threshold will be considered as correct.

    Args:
        trans: [B, P, 3]
        rot: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        contact_points: [B, P, P, 4], pairwise contact matrix.
            First item is 1 --> two parts are connecting, 0 otherwise.
            Last three items are the contacting point coordinate.

    Returns:
        [B], accuracy per data in the batch
    """
    B, P, _ = trans.shape
    thre = 0.01
    # get torch.Tensor of rotation for simplicity
    rot_type = rot.rot_type
    rot = rot.rot

    def get_min_l2_dist(points1, points2, trans1, trans2, rot1, rot2):
        """Compute the min L2 distance between two set of points."""
        # points1/2: [num_contact, num_symmetry, 3]
        # trans/rot: [num_contact, 3/4/(3, 3)]
        points1 = transform_pc(trans1, rot1, points1, rot_type=rot_type)
        points2 = transform_pc(trans2, rot2, points2, rot_type=rot_type)
        dist = ((points1[:, :, None] - points2[:, None, :]) ** 2).sum(-1)
        return dist.min(-1)[0].min(-1)[0]  # [num_contact]

    # find all contact points
    mask = contact_points[..., 0] == 1  # [B, P, P]
    # points1 = contact_points[mask][..., 1:]
    # TODO: more efficient way of getting paired contact points?
    points1, points2, trans1, trans2, rot1, rot2 = [], [], [], [], [], []
    for b in range(B):
        for i in range(P):
            for j in range(P):
                if mask[b, i, j]:
                    points1.append(contact_points[b, i, j, 1:])
                    points2.append(contact_points[b, j, i, 1:])
                    trans1.append(trans[b, i])
                    trans2.append(trans[b, j])
                    rot1.append(rot[b, i])
                    rot2.append(rot[b, j])
    points1 = torch.stack(points1, dim=0)  # [n, 3]
    points2 = torch.stack(points2, dim=0)  # [n, 3]
    # [n, 3/4/(3, 3)], corresponding translation and rotation
    trans1, trans2 = torch.stack(trans1, dim=0), torch.stack(trans2, dim=0)
    rot1, rot2 = torch.stack(rot1, dim=0), torch.stack(rot2, dim=0)
    points1 = torch.stack(get_sym_point_list(points1), dim=1)  # [n, sym, 3]
    points2 = torch.stack(get_sym_point_list(points2), dim=1)  # [n, sym, 3]
    dist = get_min_l2_dist(points1, points2, trans1, trans2, rot1, rot2)
    acc = (dist < thre).sum().float() / float(dist.numel())

    # the official code is doing avg per-contact_point acc (not per-shape)
    # so we tile the `acc` to [B]
    acc = torch.ones(B).type_as(trans) * acc
    return acc


def get_sym_point(point, x, y, z):
    """Get the symmetry point along one or many of xyz axis."""
    point = copy.deepcopy(point)
    if x == 1:
        point[..., 0] = -point[..., 0]
    if y == 1:
        point[..., 1] = -point[..., 1]
    if z == 1:
        point[..., 2] = -point[..., 2]
    return point


def get_sym_point_list(point, sym=None):
    """Get all poissible symmetry point as a list.
    `sym` is a list indicating the symmetry axis of point.
    """
    if sym is None:
        sym = [1, 1, 1]
    else:
        if not isinstance(sym, (list, tuple)):
            sym = sym.tolist()
        sym = [int(i) for i in sym]
    point_list = []
    for x in range(sym[0] + 1):
        for y in range(sym[1] + 1):
            for z in range(sym[2] + 1):
                point_list.append(get_sym_point(point, x, y, z))

    return point_list


@torch.no_grad()
def trans_metrics(trans1, trans2, valids, metric):
    """Evaluation metrics for transformation.

    Metrics used in the NSM paper.

    Args:
        trans1: [B, P, 3]
        trans2: [B, P, 3]
        valids: [B, P], 1 for input parts, 0 for padded parts
        metric: str, 'mse', 'rmse' or 'mae'

    Returns:
        [B], metric per data in the batch
    """
    assert metric in ["mse", "rmse", "mae"]
    if metric == "mse":
        metric_per_data = (trans1 - trans2).pow(2).mean(dim=-1)  # [B, P]
    elif metric == "rmse":
        metric_per_data = (trans1 - trans2).pow(2).mean(dim=-1) ** 0.5
    else:
        metric_per_data = (trans1 - trans2).abs().mean(dim=-1)
    metric_per_data = _valid_mean(metric_per_data, valids)
    return metric_per_data


@torch.no_grad()
def rot_metrics(rot1, rot2, valids, metric):
    """Evaluation metrics for rotation in euler angle (degree) space.

    Metrics used in the NSM paper.

    Args:
        rot1: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        rot2: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        valids: [B, P], 1 for input parts, 0 for padded parts
        metric: str, 'mse', 'rmse' or 'mae'

    Returns:
        [B], metric per data in the batch
    """
    assert metric in ["mse", "rmse", "mae"]
    deg1 = rot1.to_euler(to_degree=True)  # [B, P, 3]
    deg2 = rot2.to_euler(to_degree=True)
    diff1 = (deg1 - deg2).abs()
    diff2 = 360.0 - (deg1 - deg2).abs()
    # since euler angle has the discontinuity at 180
    diff = torch.minimum(diff1, diff2)
    if metric == "mse":
        metric_per_data = diff.pow(2).mean(dim=-1)  # [B, P]
    elif metric == "rmse":
        metric_per_data = diff.pow(2).mean(dim=-1) ** 0.5
    else:
        metric_per_data = diff.abs().mean(dim=-1)
    metric_per_data = _valid_mean(metric_per_data, valids)
    return metric_per_data
