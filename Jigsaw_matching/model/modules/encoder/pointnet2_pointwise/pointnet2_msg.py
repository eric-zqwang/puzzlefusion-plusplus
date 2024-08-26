import torch
import torch.nn as nn

from .pointnet2_dynamic_utils import PointNetFeaturePropagationDynamic, PointNetSetAbstractionMsgDynamic
from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation


class PointNet2PTMSG(nn.Module):
    """
    PointNet++ with multi-scale grouping
    """
    def __init__(self, feat_in=3, feat_out=2):
        super().__init__()
        self.feat_in = feat_in
        self.feat_out = feat_out
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], self.feat_in, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32 + 64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128 + 128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256 + 256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, self.feat_out, 1)
        self.bn1 = nn.BatchNorm1d(self.feat_out)

    def forward(self, x):
        """
        Input: shape(x) = [B, N, 3]
        """
        x = x.permute(0, 2, 1)
        l0_points = x
        l0_xyz = x[:, :3, :]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        x = self.bn1(self.conv1(l0_points))
        x = x.permute(0, 2, 1)
        return x


class PointNet2PTMSGDynamic(nn.Module):
    """PointNet++ with multi-scale grouping, allowing dynamic point numbers in each batch."""
    def __init__(self, feat_in=3, feat_out=2):
        super().__init__()
        self.feat_in = feat_in
        self.feat_out = feat_out
        self.sa1 = PointNetSetAbstractionMsgDynamic(0.15, [0.05, 0.1], [16, 32], self.feat_in,
                                                    [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsgDynamic(0.25, [0.1, 0.2], [16, 32], 32 + 64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsgDynamic(0.25, [0.2, 0.4], [16, 32], 128 + 128,
                                                    [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsgDynamic(0.25, [0.4, 0.8], [16, 32], 256 + 256,
                                                    [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagationDynamic(512 + 512 + 256 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagationDynamic(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagationDynamic(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagationDynamic(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, self.feat_out, 1)

    def forward(self, x, batch_length):
        """
        Input: x = [N_sum, 3]
            batch_length = [B]
        """
        idx = 0
        piece_idx = []
        for i in range(len(batch_length)):
            piece_idx.append(idx * torch.ones(batch_length[i], dtype=torch.int64).to(x.device))
            idx += 1
        piece_idx = torch.cat(piece_idx).reshape(1, 1, -1)

        l0_points = x.unsqueeze(0).permute(0, 2, 1)
        l0_xyz = x[:, :3].unsqueeze(0).permute(0, 2, 1)
        l0_piece_id = piece_idx
        l1_xyz, l1_piece_id, l1_points = self.sa1(l0_xyz, l0_piece_id, l0_points)
        l2_xyz, l2_piece_id, l2_points = self.sa2(l1_xyz, l1_piece_id, l1_points)
        l3_xyz, l3_piece_id, l3_points = self.sa3(l2_xyz, l2_piece_id, l2_points)
        l4_xyz, l4_piece_id, l4_points = self.sa4(l3_xyz, l3_piece_id, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_piece_id, l4_piece_id, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_piece_id, l3_piece_id, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_piece_id, l2_piece_id, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_piece_id, l1_piece_id, None, l1_points)

        x_out = self.conv1(l0_points)
        x_out = x_out.permute(0, 2, 1)
        return x_out.squeeze(0)


if __name__ == '__main__':
    xx = torch.randn(100, 3)
    bl = torch.tensor([20, 20, 35, 25], dtype=torch.long).reshape(4, 1)
    model_point = PointNet2PTMSGDynamic(3, 128)
    feat_point = model_point(xx, bl)
    print(feat_point.shape)
