import torch.nn as nn
import torch.nn.functional as F
from utils.pn2_utils import PointNetSetAbstraction
from chamferdist import ChamferDistance


class PN2(nn.Module):
    def __init__(self, cfg):
        super(PN2, self).__init__()
        in_channel = 3
        self.num_point = cfg.ae.num_point
        self.num_dim = cfg.ae.num_dim

        self.local_decode_pts = cfg.ae.local_decode_pts

        self.sa1 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=self.num_point, radius=0.8, nsample=64, in_channel=256 + 3, mlp=[256, 256, 512], group_all=False)
        
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=self.num_dim, kernel_size=1)
        
        self.fc1 = nn.Linear(self.num_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, self.local_decode_pts*3)

        self.loss_func = ChamferDistance()


    def forward(self, data_dict):
        # B, C, L
        xyz = data_dict["part_pcs"].permute(0, 2, 1)

        
        B, _, _ = xyz.shape
        norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)  
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)    # [B, C, L]

        global_feat = self.conv6(l3_points) # points x dim
        global_feat = global_feat.permute(0,2,1)  
        # x = global_feat.view(B, self.num_point * self.num_dim)

        x = F.relu(((self.fc1(global_feat))))
        x = F.relu((self.fc2(x)))
        x = self.fc3(x).reshape(B, self.num_point, self.local_decode_pts, 3)

        output_dict = {
            "pc_offset": x, 
            "global_feat": global_feat,
            "xyz": l3_xyz.permute(0, 2, 1)
        }

        return output_dict


    def encode(self, xyz):
        # B, C, L
        B, _, _ = xyz.shape
        norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)  
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)    # [B, C, L]

        global_feat = self.conv6(l3_points) # points x dim
        global_feat = global_feat.permute(0,2,1)  

        return global_feat, l3_xyz.permute(0, 2, 1)   # [N, L, C]


    def decode(self, global_feat):
        """
        Input: [B, L, C]
        """
        B, L, C = global_feat.shape
        x = F.relu((self.fc1(global_feat)))
        x = F.relu((self.fc2(x)))
        x = self.fc3(x).reshape(B, self.num_point, self.local_decode_pts, 3)
        
        return x


    def loss(self, data_dict, output_dict):
        pc_offset = output_dict["pc_offset"]
        xyz = output_dict["xyz"]

        pc_recon = pc_offset + xyz.unsqueeze(2)
        pc_recon = pc_recon.reshape(-1, 1000, 3)
        
        part_pcs = data_dict["part_pcs"]

        loss = self.loss_func(pc_recon, part_pcs, bidirectional=True)

        loss_dict = {"cd_loss": loss}

        return loss_dict
    
