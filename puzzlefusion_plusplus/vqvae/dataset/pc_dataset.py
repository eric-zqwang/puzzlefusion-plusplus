""" Use pre-processed point cloud data for training. """

import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import copy

class GeometryPartDataset(Dataset):
    """Geometry part assembly dataset.

    We follow the data prepared by Breaking Bad dataset:
        https://breaking-bad-dataset.github.io/
    """

    def __init__(
        self,
        cfg,
        data_dir,
        data_fn,
        category='',
        rot_range=-1,
        overfit=-1,
    ):
        self.cfg = cfg
        self.category = category if category.lower() != 'all' else ''

        self.data_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npz')])

        self.max_num_part = cfg.data.max_num_part
        self.min_num_part = cfg.data.min_num_part

        if overfit != -1: 
            self.data_files = self.data_files[:overfit] 
        
        self.data_list = []
        self.rot_range = rot_range

        for file_name in tqdm(self.data_files):
            data_dict = np.load(os.path.join(self.data_dir, file_name))

            pc = data_dict['part_pcs_gt']
            data_id = data_dict['data_id'].item()
            part_valids = data_dict['part_valids']
            num_parts = data_dict["num_parts"].item()
            mesh_file_path = data_dict['mesh_file_path'].item()
            category = data_dict["category"]
            
            sample = {
                'part_pcs': pc,
                'data_id': data_id,
                'part_valids': part_valids,
                'mesh_file_path': mesh_file_path,
                'num_parts': num_parts,
            }

            if num_parts > self.max_num_part or num_parts < self.min_num_part:
                continue

            self.data_list.append(sample)

    @staticmethod
    def _recenter_pc(pc):
        """pc: [N, 3]"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid[None]
        return pc, centroid

    @staticmethod
    def _rotate_pc(pc):
        """pc: [N, 3]"""
        rot_mat = R.random().as_matrix()
        pc = (rot_mat @ pc.T).T
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        # we use scalar-first quaternion
        quat_gt = quat_gt[[3, 0, 1, 2]]
        return pc, quat_gt


    def _pad_data(self, data):
        """Pad data to shape [`self.max_num_part`, data.shape[1], ...]."""
        data = np.array(data)
        pad_shape = (self.max_num_part, ) + tuple(data.shape[1:])
        pad_data = np.zeros(pad_shape, dtype=np.float32)
        pad_data[:data.shape[0]] = data
        return pad_data


    def __getitem__(self, idx):
        """
        recenter the fragments, and random rotate it to train ae
        """
        data_dict = copy.deepcopy(self.data_list[idx])
        pcs = data_dict['part_pcs']
        num_parts = data_dict['num_parts']

        cur_pts = []
        for i in range(num_parts):
            pc = pcs[i]
            pc, _ = self._recenter_pc(pc)
            pc, _ = self._rotate_pc(pc)
            cur_pts.append(pc)
            
        cur_pts = self._pad_data(np.stack(cur_pts, axis=0))  # [P, N, 3]
        scale = np.max(np.abs(cur_pts), axis=(1,2), keepdims=True)
        scale[scale == 0] = 1
        cur_pts = cur_pts / scale

        data_dict['part_pcs'] = cur_pts
        
        return data_dict

    def __len__(self):
        return len(self.data_list)


def build_geometry_dataloader(cfg):
    data_dict = dict(
        cfg=cfg,
        data_dir=cfg.data.data_dir,
        data_fn='train',
        category=cfg.data.category,
        rot_range=cfg.data.rot_range,
        overfit=cfg.data.overfit,
    )
    train_set = GeometryPartDataset(**data_dict)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.data.num_workers > 0),
    )

    data_dict['data_fn'] = 'val'
    data_dict['data_dir'] = cfg.data.data_val_dir
    val_set = GeometryPartDataset(**data_dict)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.data.num_workers > 0),
    )
    return train_loader, val_loader
