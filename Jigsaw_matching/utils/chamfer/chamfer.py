# import chamfer_cuda
import torch
from torch.cuda.amp import custom_fwd, custom_bwd


def safe_sqrt(x, eps=1e-12):
    return torch.sqrt(torch.clamp(x, eps))


class ChamferDistanceFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."

        dist1, idx1, dist2, idx2 = chamfer_cuda.chamfer_forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_dist1, grad_dist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_dist1 = grad_dist1.contiguous()
        grad_dist2 = grad_dist2.contiguous()
        assert grad_dist1.is_cuda and grad_dist2.is_cuda, "Only support cuda currently."
        grad_xyz1, grad_xyz2 = chamfer_cuda.chamfer_backward(
            grad_dist1, grad_dist2, xyz1, xyz2, idx1, idx2)
        return grad_xyz1, grad_xyz2


def chamfer_distance(xyz1, xyz2, transpose=False, sqrt=False, eps=1e-12):
    """Chamfer distance

    Args:
        xyz1 (torch.Tensor): (b, n1, 3)
        xyz2 (torch.Tensor): (b, n1, 3)
        transpose (bool): whether to transpose inputs as it might be BCN format.
            Extensions only support BNC format.
        sqrt (bool): whether to square root distance
        eps (float): to safely sqrt

    Returns:
        dist1 (torch.Tensor): (b, n1)
        dist2 (torch.Tensor): (b, n2)

    """
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)

    if transpose:
        xyz1 = xyz1.transpose(1, 2)
        xyz2 = xyz2.transpose(1, 2)
    xyz1, xyz2 = xyz1.to(torch.double), xyz2.to(torch.double)
    dist1, dist2 = ChamferDistanceFunction.apply(xyz1, xyz2)
    if sqrt:
        dist1 = safe_sqrt(dist1, eps)
        dist2 = safe_sqrt(dist2, eps)
    return dist1, dist2


def nn_distance(xyz1, xyz2, transpose=True):
    """The interface to infer rather than train"""
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)
    if transpose:
        xyz1 = xyz1.transpose(1, 2).contiguous()
        xyz2 = xyz2.transpose(1, 2).contiguous()
    return chamfer_cuda.chamfer_forward(xyz1, xyz2)


class ChamferDistanceL2(torch.nn.Module):
    f''' Chamder Distance L2
    '''

    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferDistanceFunction.apply(xyz1, xyz2)
        return torch.mean(dist1) + torch.mean(dist2)


class ChamferDistanceL2_split(torch.nn.Module):
    f""" Chamder Distance L2
    """

    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferDistanceFunction.apply(xyz1, xyz2)
        return torch.mean(dist1), torch.mean(dist2)


class ChamferDistanceL1(torch.nn.Module):
    f""" Chamder Distance L1
    """

    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferDistanceFunction.apply(xyz1, xyz2)
        # import pdb
        # pdb.set_trace()
        dist1 = torch.sqrt(dist1)
        dist2 = torch.sqrt(dist2)
        return (torch.mean(dist1) + torch.mean(dist2)) / 2


class ChamferDistanceL1_PM(torch.nn.Module):
    f""" Chamder Distance L1
    """

    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, _ = ChamferDistanceFunction.apply(xyz1, xyz2)
        dist1 = torch.sqrt(dist1)
        return torch.mean(dist1)
