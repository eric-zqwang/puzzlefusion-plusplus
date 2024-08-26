import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool import fps, knn
from torch_geometric.utils import to_dense_batch


def square_distance_with_piece(src, dst, src_piece_id, dst_piece_id):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
        src_piece_id: piece id of source points, [B, N, 1]
        dst_piece_id: piece id of target points, [B, M, 1]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    device = src.device
    piece_dist = torch.ones(B, N, M).to(device)
    indices = torch.where(src_piece_id.repeat(1, 1, M) == dst_piece_id.view(B, 1, M).repeat(1, N, 1))
    piece_dist[indices] *= 0
    dist += piece_dist * 1e6

    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz, piece_id, new_piece_id):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
        piece_id: piece indices of all points, [B, N, 1]
        new_piece_id: piece indices of query points, [B, S, 1]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance_with_piece(new_xyz, xyz, new_piece_id, piece_id)
    # sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    if group_idx.shape[-1] < nsample:
        group_idx = torch.cat([group_idx, N * torch.ones(size=[B, S, nsample - N], dtype=torch.long).to(device)],
                              dim=-1)
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class PointNetSetAbstractionMsgDynamic(nn.Module):
    def __init__(self, ratio, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsgDynamic, self).__init__()
        self.ratio = ratio
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, piece_id, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            piece_id: input piece index of points, [B, 1, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        piece_id = piece_id.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape

        assert B == 1
        centroids = fps(xyz[0, :, :], batch=piece_id.reshape(B * N), ratio=self.ratio).unsqueeze(0)
        S = centroids.shape[1]
        new_xyz = index_points(xyz, centroids)
        new_piece_id = index_points(piece_id, centroids)
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]

            group_idx = knn(xyz[0, :, :], new_xyz[0, :, :], k=K, batch_x=piece_id.reshape(-1),
                            batch_y=new_piece_id.reshape(-1))
            group_idx = to_dense_batch(group_idx[1], group_idx[0], fill_value=N, max_num_nodes=K)[0].unsqueeze(0)
            group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, K])
            mask = group_idx == N
            group_idx[mask] = group_first[mask]

            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_piece_id = new_piece_id.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_piece_id, new_points_concat


class PointNetFeaturePropagationDynamic(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagationDynamic, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, piece_id1, piece_id2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            piece_id1: piece index of input points, [B, N]
            piece_id2: piece index of sampled input points, [B, S]
            points1: input points data, [B, D, N]
            points2: sampled input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        piece_id1 = piece_id1.permute(0, 2, 1)
        piece_id2 = piece_id2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # dists = square_distance_with_piece(xyz1, xyz2, piece_id1, piece_id2)
            # dists, idx = dists.sort(dim=-1)
            # dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            group_idx = knn(xyz2[0, :, :], xyz1[0, :, :], k=3, batch_x=piece_id2.reshape(-1),
                            batch_y=piece_id1.reshape(-1))
            dists = xyz1[:, group_idx[0], :] ** 2 + xyz2[:, group_idx[1], :] ** 2
            dists -= 2 * xyz1[:, group_idx[0], :] * xyz2[:, group_idx[1], :]
            dists = torch.sum(dists, dim=-1).reshape(-1)
            idx = to_dense_batch(group_idx[1], group_idx[0], fill_value=-1, max_num_nodes=3)[0].unsqueeze(0)
            idx_first = idx[:, :, 0].view(B, xyz1.shape[1], 1).repeat([1, 1, 3])
            mask = idx == -1
            idx[mask] = idx_first[mask]
            dists = to_dense_batch(dists, group_idx[0], fill_value=1e8, max_num_nodes=3)[0].unsqueeze(0)

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
