import torch
from puzzlefusion_plusplus.denoiser.evaluation.transform import (
    transform_pc,
    quaternion_to_euler,
)


def _valid_mean(loss_per_part, valids):
    """Average loss values according to the valid parts.

    Args:
        loss_per_part: [B, P]
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch, averaged over valid parts
    """
    nan_mask = torch.isnan(loss_per_part)
    loss_per_part[nan_mask] = 0.
    valids = valids.float().detach()
    loss_per_data = (loss_per_part * valids).sum(1) / valids.sum(1)
    return loss_per_data


def trans_metrics(trans1, trans2, valids, metric):
    """Evaluation metrics for translation.

    Metrics used in the NSM paper.

    Args:
        trans1: [B, P, 3], pred translation
        trans2: [B, P, 3], gt translation
        valids: [B, P], 1 for input parts, 0 for padded parts
        metric: str, 'mse', 'rmse' or 'mae'

    Returns:
        [B], metric per data in the batch
    """
    assert metric in ['mse', 'rmse', 'mae']
    if metric == 'mse':
        metric_per_data = (trans1 - trans2).pow(2).mean(dim=-1)  # [B, P]
    elif metric == 'rmse':
        metric_per_data = (trans1 - trans2).pow(2).mean(dim=-1)**0.5
    else:
        metric_per_data = (trans1 - trans2).abs().mean(dim=-1)
    metric_per_data = _valid_mean(metric_per_data, valids)
    return metric_per_data


@torch.no_grad()
def rot_metrics(rot1, rot2, valids, metric):
    """Evaluation metrics for rotation in euler angle (degree) space.

    Metrics used in the NSM paper.

    Args:
        rot1: [B, P, 4], pred quat
        rot2: [B, P, 4], gt quat
        valids: [B, P], 1 for input parts, 0 for padded parts
        metric: str, 'mse', 'rmse' or 'mae'

    Returns:
        [B], metric per data in the batch
    """
    assert metric in ['mse', 'rmse', 'mae']
    deg1 = quaternion_to_euler(rot1, to_degree=True)  # [B, P, 3]
    deg2 = quaternion_to_euler(rot2, to_degree=True)

    diff1 = (deg1 - deg2).abs()
    diff2 = 360. - (deg1 - deg2).abs()
    # since euler angle has the discontinuity at 180
    diff = torch.minimum(diff1, diff2)
    if metric == 'mse':
        metric_per_data = diff.pow(2).mean(dim=-1)  # [B, P]
    elif metric == 'rmse':
        metric_per_data = diff.pow(2).mean(dim=-1)**0.5
    else:
        metric_per_data = diff.abs().mean(dim=-1)
    metric_per_data = _valid_mean(metric_per_data, valids)
    return metric_per_data


@torch.no_grad()
def calc_part_acc(pts, trans1, trans2, rot1, rot2, valids, chamfer_distance):
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
    return acc, acc_per_part, loss_per_data


@torch.no_grad()
def calc_shape_cd(pts, trans1, trans2, rot1, rot2, valids, chamfer_distance):
    
    B, P, N, _ = pts.shape
    
    valid_mask = valids[..., None, None]  # [B, P, 1, 1]
    
    pts = pts.detach().clone()
    
    pts = pts.masked_fill(valid_mask == 0, 1e3)
    
    pts1 = transform_pc(trans1, rot1, pts)  # [B, P, N, 3]
    pts2 = transform_pc(trans2, rot2, pts)
    
    shape1 = pts1.flatten(1, 2)
    shape2 = pts2.flatten(1, 2)
    
    shape_cd = chamfer_distance(
        shape1, 
        shape2, 
        bidirectional=True, 
        point_reduction=None, 
        batch_reduction=None
    )
    
    shape_cd = shape_cd.view(B, P, N).mean(-1)
    shape_cd = _valid_mean(shape_cd, valids)
    
    return shape_cd
      


        