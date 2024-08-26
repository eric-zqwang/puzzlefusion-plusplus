import copy
import itertools
import pickle

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from torch.nn import LayerNorm, GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm


def pickle_load(file, **kwargs):
    if isinstance(file, str):
        with open(file, "rb") as f:
            obj = pickle.load(f, **kwargs)
    elif hasattr(file, "read"):
        obj = pickle.load(file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
    return obj


def pickle_dump(obj, file=None, **kwargs):
    kwargs.setdefault("protocol", 2)
    if file is None:
        return pickle.dumps(obj, **kwargs)
    elif isinstance(file, str):
        with open(file, "wb") as f:
            pickle.dump(obj, f, **kwargs)
    elif hasattr(file, "write"):
        pickle.dump(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def save_pc(pc, file):
    """Save point cloud to file.

    Args:
        pc (np.ndarray): [N, 3]
        file (str)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.io.write_point_cloud(file, pcd)


def colorize_part_pc(part_pc, colors):
    """Colorize part point cloud.

    Args:
        part_pc (np.ndarray): [P, N, 3]
        colors (np.ndarray): [max_num_parts, 3 (RGB)]

    Returns:
        np.ndarray: [P, N, 6]
    """
    P, N, _ = part_pc.shape
    colored_pc = np.zeros((P, N, 6))
    colored_pc[:, :, :3] = part_pc
    for i in range(P):
        colored_pc[i, :, 3:] = colors[i]
    return colored_pc


def array_equal(a, b):
    """Compare if two arrays are the same.

    Args:
        a/b: can be np.ndarray or torch.Tensor.
    """
    if a.shape != b.shape:
        return False
    try:
        assert (a == b).all()
        return True
    except:
        return False


def array_in_list(array, lst):
    """Judge whether an array is in a list."""
    for v in lst:
        if array_equal(array, v):
            return True
    return False


def filter_wd_parameters(model, skip_list=()):
    """Create parameter groups for optimizer.

    We do two things:
        - filter out params that do not require grad
        - exclude bias and Norm layers
    """
    # we need to sort the names so that we can save/load ckps
    w_name, b_name, no_decay_name = [], [], []
    for name, m in model.named_modules():
        # exclude norm weight
        if isinstance(m, (LayerNorm, GroupNorm, _BatchNorm, _InstanceNorm)):
            w_name.append(name)
        # exclude bias
        if hasattr(m, "bias") and m.bias is not None:
            b_name.append(name)
        if name in skip_list:
            no_decay_name.append(name)
    w_name.sort()
    b_name.sort()
    no_decay_name.sort()
    no_decay = [model.get_submodule(m).weight for m in w_name] + [
        model.get_submodule(m).bias for m in b_name
    ]
    for name in no_decay_name:
        no_decay += [
            p
            for p in model.get_submodule(m).parameters()
            if p.requires_grad and not array_in_list(p, no_decay)
        ]

    decay_name = []
    for name, param in model.named_parameters():
        if param.requires_grad and not array_in_list(param, no_decay):
            decay_name.append(name)
    decay_name.sort()
    decay = [model.get_parameter(name) for name in decay_name]
    return {"decay": list(decay), "no_decay": list(no_decay)}


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def lexico_iter(lex):
    return itertools.combinations(lex, 2)


def match_mat_to_piecewise(perm_mat, n_critical_pcs, n_p=None, transposed=None):
    """
    matching mat to piecewise matching matrix
    :param perm_mat: [B, N_sum, N_sum]
    :param n_critical_pcs: [B, P]
    :param n_p: [B]
    :param transposed: [B]
    :return: [B, P, P, N', N']
    """
    if isinstance(perm_mat, torch.Tensor):
        N_ = torch.max(n_critical_pcs)
        B, P = n_critical_pcs.shape
        match_mat = torch.zeros(B, P, P, N_, N_, device=perm_mat.device)
        n_sum = torch.cumsum(n_critical_pcs, dim=-1)
        if P == 2 and perm_mat.shape[1] < torch.max(n_sum):
            if transposed is None:
                transposed = torch.zeros(B, dtype=torch.bool)
            for b in range(B):
                n1, n2 = n_critical_pcs[b, 0], n_critical_pcs[b, 1]
                if transposed[b]:
                    match_mat[b, 0, 1, :n1, :n2] = perm_mat[
                                                   b, :n2, :n1
                                                   ].transpose(1, 0)
                    match_mat[b, 1, 0, :n2, :n1] = perm_mat[b, :n2, :n1]
                else:
                    match_mat[b, 0, 1, :n1, :n2] = perm_mat[b, :n1, :n2]
                    match_mat[b, 1, 0, :n2, :n1] = perm_mat[
                                                   b, :n1, :n2
                                                   ].transpose(1, 0)
        else:
            if n_p is None:
                n_p = torch.ones(B, dtype=torch.long) * P
            for b in range(B):
                for p1, p2 in lexico_iter(torch.arange(n_p[b])):
                    if n_critical_pcs[b, p1] == 0 or n_critical_pcs[b, p2] == 0:
                        continue
                    lp1 = 0 if p1 == 0 else n_sum[b, p1 - 1]
                    lp2 = 0 if p2 == 0 else n_sum[b, p2 - 1]
                    match_mat[
                    b,
                    p1,
                    p2,
                    : n_critical_pcs[b, p1],
                    : n_critical_pcs[b, p2],
                    ] = perm_mat[b, lp1: n_sum[b, p1], lp2: n_sum[b, p2]]
                    match_mat[
                    b,
                    p2,
                    p1,
                    : n_critical_pcs[b, p2],
                    : n_critical_pcs[b, p1],
                    ] = perm_mat[b, lp2: n_sum[b, p2], lp1: n_sum[b, p1]]
        return match_mat
    elif isinstance(perm_mat, np.ndarray):
        N_ = np.max(n_critical_pcs)
        B, P = n_critical_pcs.shape
        match_mat = np.zeros([B, P, P, N_, N_])
        n_sum = np.cumsum(n_critical_pcs, axis=-1)
        if P == 2 and perm_mat.shape[1] < np.max(n_sum):
            if transposed is None:
                transposed = np.zeros(B, dtype=np.bool)
            for b in range(B):
                n1, n2 = n_critical_pcs[b, 0], n_critical_pcs[b, 1]
                if transposed[b]:
                    match_mat[b, 0, 1, :n1, :n2] = perm_mat[
                                                   b, :n2, :n1
                                                   ].transpose(1, 0)
                    match_mat[b, 1, 0, :n2, :n1] = perm_mat[b, :n2, :n1]
                else:
                    match_mat[b, 0, 1, :n1, :n2] = perm_mat[b, :n1, :n2]
                    match_mat[b, 1, 0, :n2, :n1] = perm_mat[
                                                   b, :n1, :n2
                                                   ].transpose(1, 0)
        else:
            if n_p is None:
                n_p = np.ones(B, dtype=np.int) * P
            for b in range(B):
                for p1, p2 in lexico_iter(np.arange(n_p[b])):
                    if n_critical_pcs[b, p1] == 0 or n_critical_pcs[b, p2] == 0:
                        continue
                    lp1 = 0 if p1 == 0 else n_sum[b, p1 - 1]
                    lp2 = 0 if p2 == 0 else n_sum[b, p2 - 1]
                    match_mat[
                    b,
                    p1,
                    p2,
                    : n_critical_pcs[b, p1],
                    : n_critical_pcs[b, p2],
                    ] = perm_mat[b, lp1: n_sum[b, p1], lp2: n_sum[b, p2]]
                    match_mat[
                    b,
                    p2,
                    p1,
                    : n_critical_pcs[b, p2],
                    : n_critical_pcs[b, p1],
                    ] = perm_mat[b, lp2: n_sum[b, p2], lp1: n_sum[b, p1]]
        return match_mat


def get_batch_length_from_part_points(n_pcs, n_valids=None, part_valids=None):
    """
    :param n_pcs: [B, P] number of points per batch
    :param n_valids: [B] number of parts per batch
    :param part_valids: [B, P] 0/1
    :return: batch_length [\sum n_valids, ]
    """
    B, P = n_pcs.shape
    if n_valids is None:
        if part_valids is None:
            n_valids = torch.ones(B, device=n_pcs, dtype=torch.long) * P
        else:
            n_valids = torch.sum(part_valids, dim=1).to(torch.long)

    batch_length_list = []
    for b in range(B):
        batch_length_list.append(n_pcs[b, : n_valids[b]])
    batch_length = torch.cat(batch_length_list)
    assert batch_length.shape[0] == torch.sum(n_valids)
    return batch_length
