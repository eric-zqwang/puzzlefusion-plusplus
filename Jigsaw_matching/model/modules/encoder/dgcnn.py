"""Adopted and modified from https://github.com/AnTao97/dgcnn.pytorch/blob/master/model.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric


def knn(x, k):
    """x: [B, C, N]"""
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (B, N, k)
    return idx


def get_graph_feature(x, k=20):
    """x: [B, C, N]"""
    idx = knn(x, k=k)   # (batch_size, num_points, k)

    batch_size, num_dims, num_points = x.size()
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = (idx + idx_base).view(-1)

    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    # batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature      # (batch_size, 2*num_dims, num_points, k)


def get_graph_feature_dynamic(x, k=20, batch_x=None):
    """x: [N_sum, C]"""
    device = x.device
    N, n_dim = x.shape
    if batch_x is None:
        batch_x = torch.zeros(N, dtype=torch.long, device=device)
    idx = torch_geometric.nn.knn(x, x, k, batch_x, batch_x)
    idx, msk = torch_geometric.utils.to_dense_batch(idx[1], idx[0], fill_value=N, max_num_nodes=k)
    x = torch.cat([x, torch.zeros(1, n_dim).to(device)], dim=0)
    feature = x[idx.view(-1).long(), :]
    feature = feature.view(N, k, n_dim)  # [N, k, dim]
    x = x[:N].view(N, 1, n_dim).repeat(1, k, 1)
    msk = msk.to(torch.float32).unsqueeze(-1)
    feature = torch.cat((feature - x, x), dim=-1) * msk
    feature = feature.permute(0, 2, 1).contiguous()
    return feature      # (N_sum, 2*num_dims, k)


class DGCNN(nn.Module):
    """DGCNN feature extractor.

    Input point clouds [B, N, 3].
    Output per-point feature [B, N, feat_dim] or global feature [B, feat_dim].
    """

    def __init__(self, feat_dim, global_feat=True):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(feat_dim)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, feat_dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.global_feat = global_feat
        if global_feat:
            self.out_fc = nn.Linear(feat_dim * 2, feat_dim)

    def forward(self, x):
        """x: [B, N, 3]"""
        x = x.transpose(2, 1).contiguous()  # [B, 3, N]
        batch_size = x.size(0)
        x = get_graph_feature(x)   # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)          # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1)[0]      # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)          # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1)[0]      # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)          # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1)[0]      # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)          # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1)[0]      # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)          # (batch_size, 64+64+128+256, num_points) -> (batch_size, feat_dim, num_points)

        if self.global_feat:
            x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # (batch_size, feat_dim, num_points) -> (batch_size, feat_dim)
            x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # (batch_size, feat_dim, num_points) -> (batch_size, feat_dim)
            x = torch.cat((x1, x2), 1)              # (batch_size, feat_dim*2)
            feat = self.out_fc(x)  # [B, feat_dim]
        else:
            feat = x.transpose(2, 1).contiguous()  # [B, N, feat_dim]

        return feat


class DGCNNDynamic(nn.Module):
    """DGCNN feature extractor.

    Input point clouds [N_sum, 3].
    Output per-point feature [N_sum, feat_dim] or global feature [B, feat_dim].
    """

    def __init__(self, feat_dim, global_feat=True, in_feat_dim=3):
        super().__init__()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(feat_dim)

        self.conv1 = nn.Sequential(nn.Conv1d(in_feat_dim*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, feat_dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.global_feat = global_feat
        if global_feat:
            self.out_fc = nn.Linear(feat_dim * 2, feat_dim)

    def forward(self, x, batch_length):
        """
            x: [N_sum, 3],
            batch_length [B]"""
        batch_length = batch_length.reshape(-1)
        B = batch_length.shape[0]
        batch_x = torch.tensor([b for b in range(batch_length.shape[0]) for t in range(int(batch_length[b]))],
                               dtype=torch.long, device=x.device)

        x = get_graph_feature_dynamic(x, batch_x=batch_x)  # [N_sum, 3] -> [N_sum, 3*2, k]
        x = self.conv1(x)          # N, 6, k -> N, 64, k
        x1 = x.max(dim=-1)[0]      # N, 64, k -> N, 64

        x = get_graph_feature_dynamic(x1, batch_x=batch_x)  # [N_sum, 64] -> [N_sum, 64*2, k]
        x = self.conv2(x)          # N, 128, k -> N, 64, k
        x2 = x.max(dim=-1)[0]      # N, 64, k -> N, 64

        x = get_graph_feature_dynamic(x2, batch_x=batch_x)  # [N_sum, 64] -> [N_sum, 64*2, k]
        x = self.conv3(x)          # N, 128, k -> N, 128, k
        x3 = x.max(dim=-1)[0]      # N, 128, k -> N, 128

        x = get_graph_feature_dynamic(x3, batch_x=batch_x)  # [N_sum, 128] -> [N_sum, 128*2, k]
        x = self.conv4(x)          # N, 256, k -> N, 256, k
        x4 = x.max(dim=-1)[0]      # N, 256, k -> N, 256

        x = torch.cat((x1, x2, x3, x4), dim=1)  # N, 64+64+128+256

        x = x.unsqueeze(0).transpose(1, 2).contiguous()
        x = self.conv5(x)          # 1, 64+64+128+256, N -> 1, feat_dim, N

        if self.global_feat:
            batch_pos = torch.cumsum(batch_length, dim=-1, dtype=torch.long)
            x1_list = []
            x2_list = []
            for b in range(B):
                st = 0 if b == 0 else batch_pos[b-1]
                ed = batch_pos[b]
                f_b = x[:, :, st:ed]
                x1_list.append(F.adaptive_max_pool1d(f_b, 1).squeeze(-1))  # 1, feat_dim
                x2_list.append(F.adaptive_avg_pool1d(f_b, 1).squeeze(-1))
            x1 = torch.cat(x1_list, dim=0)
            x2 = torch.cat(x2_list, dim=0)  # B, feat_dim
            x = torch.cat((x1, x2), dim=-1)  # B, feat_dim * 2
            feat = self.out_fc(x)  # B, feat_dim
        else:
            feat = x.transpose(2, 1).squeeze(0).contiguous()  # [N, feat_dim]

        return feat


if __name__ == '__main__':
    xx = torch.randn(100, 3)
    bl = torch.tensor([15, 20, 35, 30], dtype=torch.long).reshape(4, 1)
    model_global = DGCNNDynamic(128, True, 3)
    model_point = DGCNNDynamic(128, False, 3)
    feat_global = model_global(xx, bl)
    feat_point = model_point(xx, bl)
    print(feat_global.shape)
    print(feat_point.shape)

