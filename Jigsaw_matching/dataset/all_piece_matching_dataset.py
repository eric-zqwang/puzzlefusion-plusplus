import os
import pickle
import random

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader


class AllPieceMatchingDataset(Dataset):
    """Geometry part assembly dataset, with fracture surface information.

    We follow the data prepared by Breaking Bad dataset:
        https://breaking-bad-dataset.github.io/
    """

    def __init__(
            self,
            data_dir,
            data_fn,
            data_keys,
            category="all",
            num_points=1000,
            min_num_part=2,
            max_num_part=20,
            shuffle_parts=False,
            rot_range=-1,
            overfit=10,
            length=-1,

            sample_by="area",
            min_part_point=30,
            fracture_label_threshold=0.025,
    ):
        # store parameters
        self.category = category if category.lower() != "all" else ""
        # self.category = "Bottle"

        self.data_dir = data_dir
        self.num_points = num_points
        self.min_num_part = min_num_part
        self.max_num_part = max_num_part  # ignore shapes with more parts
        self.min_part_point = min_part_point  # ensure that each piece has at least # points
        self.shuffle_parts = shuffle_parts  # shuffle part orders
        self.rot_range = rot_range  # rotation range in degree

        self.sample_by = sample_by
        # ['point', 'area'] sample by fixed point number or mesh area, we only support 'area' now.
        # list of fracture folder path
        self.data_list = self._read_data(data_fn)

        print("dataset length: ", len(self.data_list))

        # additional data to load, e.g. ('part_ids', 'instance_label')
        self.data_keys = data_keys

        # overfit = 10
        
        if overfit > 0:
            self.data_list = self.data_list[:overfit]

        if 0 < length < len(self.data_list):
            self.length = length
            if shuffle_parts:
                pos = list(range(len(self.data_list)))
                random.shuffle(pos)
                self.data_list = [self.data_list[i] for i in pos]
        else:
            self.length = len(self.data_list)

        self.fracture_label_threshold = fracture_label_threshold

    def __len__(self):
        return self.length

    def _read_data(self, data_fn):
        """Filter out invalid number of parts and generate data_list."""
        # Load pre-generated data_list if exists.
        pre_compute_file_name = f"all_piece_matching_metadata_{self.min_num_part}_{self.max_num_part}_" + data_fn
        if os.path.exists(os.path.join(self.data_dir, pre_compute_file_name)):
            with open(os.path.join(self.data_dir, pre_compute_file_name), "rb") as meta_table:
                meta_dict = pickle.load(meta_table)
                data_list = meta_dict["data_list"]
            return data_list

        print("start generate data_list")
        with open(os.path.join(self.data_dir, data_fn), "r") as f:
            mesh_list = [line.strip() for line in f.readlines()]
            if self.category:
                mesh_list = [line for line in mesh_list if self.category in line.split("/")]
        data_list = []
        for mesh in mesh_list:
            mesh_dir = os.path.join(self.data_dir, mesh)
            if not os.path.isdir(mesh_dir):
                print(f"{mesh} does not exist")
                continue
            fracs = os.listdir(mesh_dir)
            fracs.sort()
            for frac in fracs:
                # we take both fractures and modes for training
                if "fractured" not in frac and "mode" not in frac:
                    continue
                frac = os.path.join(mesh, frac)
                pieces = os.listdir(os.path.join(self.data_dir, frac))
                pieces.sort()
                if self.min_num_part <= len(pieces) <= self.max_num_part:
                    data_list.append(frac)
        print("finish generation, start saving")
        meta_dict = {
            "data_list": data_list,
        }
        with open(os.path.join(self.data_dir, pre_compute_file_name), "wb") as meta_table:
            pickle.dump(meta_dict, meta_table, protocol=pickle.HIGHEST_PROTOCOL)
        return data_list

    @staticmethod
    def _recenter_pc(pc):
        """pc: [N, 3]"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid[None]
        return pc, centroid

    def _rotate_pc(self, pc):
        """
        pc: [N, 3]
        normal: [N, 3]
        """

        if self.rot_range > 0.0:
            rot_euler = (np.random.rand(3) - 0.5) * 2.0 * self.rot_range
            rot_mat = R.from_euler("xyz", rot_euler, degrees=True).as_matrix()
        else:
            rot_mat = R.random().as_matrix()
        pc = (rot_mat @ pc.T).T
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        # we use scalar-first quaternion
        quat_gt = quat_gt[[3, 0, 1, 2]]
        return pc, quat_gt

    @staticmethod
    def _shuffle_pc(pc, pc_gt):
        """pc: [N, 3]"""
        order = np.arange(pc.shape[0])
        random.shuffle(order)
        pc = pc[order]
        pc_gt = pc_gt[order]
        return pc, pc_gt

    def _pad_data(self, data, pad_size=None):
        """Pad data to shape [`self.max_num_part`, data.shape[1], ...]."""
        if pad_size is None:
            pad_size = self.max_num_part
        data = np.array(data)
        if len(data.shape) > 1:
            pad_shape = (pad_size,) + tuple(data.shape[1:])
        else:
            pad_shape = (pad_size,)
        pad_data = np.zeros(pad_shape, dtype=data.dtype)
        pad_data[: data.shape[0]] = data
        return pad_data

    @staticmethod
    def sample_points_by_areas(areas, num_points):
        """areas: [P], num_points: N"""
        total_area = np.sum(areas)
        nps = np.ceil(areas * num_points / total_area).astype(np.int32)
        nps[np.argmax(nps)] -= np.sum(nps) - num_points
        return np.array(nps, dtype=np.int64)

    def sample_reweighted_points_by_areas(self, areas):
        """ Sample points by areas, but ensures that each part has at least # points.
        areas: [P]
        """
        nps = self.sample_points_by_areas(areas, self.num_points)
        if self.min_part_point <= 1:
            return nps
        delta = 0
        for i in range(len(nps)):
            if nps[i] < self.min_part_point:
                delta += self.min_part_point - nps[i]
                nps[i] = self.min_part_point
        while delta > 0:
            k = np.argmax(nps)
            if nps[k] - delta >= self.min_part_point:
                nps[k] -= delta
                delta = 0
            else:
                delta -= nps[k] - self.min_part_point
                nps[k] = self.min_part_point
        # simply take points from the largest parts
        # This implementation is not very elegant, could improve by resample by areas.
        return np.array(nps, dtype=np.int64)

    def _get_pcs(self, data_folder):
        """Read mesh and sample point cloud from a folder."""
        # `piece`: xxx/plate/1d4093ad2dfad9df24be2e4f911ee4af/fractured_0/piece_0.obj
        data_folder = os.path.join(self.data_dir, data_folder)
        mesh_files = os.listdir(data_folder)
        mesh_files.sort()
        if not self.min_num_part <= len(mesh_files) <= self.max_num_part:
            raise ValueError

        # read mesh and sample points
        meshes = [
            trimesh.load(os.path.join(data_folder, mesh_file), force="mesh")
            for mesh_file in mesh_files
        ]
        areas = [mesh.area for mesh in meshes]
        areas = np.array(areas)
        pcs, piece_id, nps = [], [], []
        if self.sample_by == "area":
            nps = self.sample_reweighted_points_by_areas(areas)
        else:
            raise NotImplementedError(f"Must sample by area")

        for i, (mesh) in enumerate(meshes):
            num_points = nps[i]
            samples, fid = mesh.sample(num_points, return_index=True)
            pcs.append(samples)
            piece_id.append([i] * num_points)

        piece_id = np.concatenate(piece_id).astype(np.int64).reshape((-1, 1))
        return pcs, piece_id, nps, areas

    def __getitem__(self, index):
        pcs, piece_id, nps, areas = self._get_pcs(self.data_list[index])
        num_parts = len(pcs)
        cur_pts, cur_quat, cur_trans, cur_pts_gt = [], [], [], []
        for i, (pc, n_p) in enumerate(zip(pcs, nps)):
            pc_gt = pc.copy()
            pc, gt_trans = self._recenter_pc(pc)
            pc, gt_quat = self._rotate_pc(pc)
            pc_shuffle, pc_gt_shuffle = self._shuffle_pc(pc, pc_gt)

            cur_pts.append(pc_shuffle)
            cur_quat.append(gt_quat)
            cur_trans.append(gt_trans)
            cur_pts_gt.append(pc_gt_shuffle)

        cur_pts = np.concatenate(cur_pts).astype(np.float32)  # [N_sum, 3]
        cur_pts_gt = np.concatenate(cur_pts_gt).astype(np.float32)  # [N_sum, 3]
        cur_quat = self._pad_data(np.stack(cur_quat, axis=0), self.max_num_part).astype(np.float32)  # [P, 4]
        cur_trans = self._pad_data(np.stack(cur_trans, axis=0), self.max_num_part).astype(np.float32)  # [P, 3]
        n_pcs = self._pad_data(np.array(nps), self.max_num_part).astype(np.int64)  # [P]
        valids = np.zeros(self.max_num_part, dtype=np.float32)
        valids[:num_parts] = 1.0
        # soft threshold, parameter not tuned.
        # threshold = 1 / np.sqrt(self.num_points)
        # label_thresholds = 2 * threshold * np.sqrt(areas)[piece_id[:, 0]]
        label_thresholds = np.ones([self.num_points], dtype=np.float32) * self.fracture_label_threshold  # [N_sum]
        """
        data_dict = {
            'part_pcs', 'gt_pcs': [P, N, 3], or [N_sum, 3], The points sampled from each part.
            'part_valids': P, 1 for shape parts, 0 for padded zeros.
            'part_quat': [P, 4], Rotation as quaternion.
            'part_trans': [P, 3], Translation vector.
            'n_pcs': [P], number of points in each piece.
            'data_id': int, ID of the data.
            'critical_label_threshold': [N_sum] thresholds for calc ground truth label.
        }
        """

        mesh_file_path = self.data_list[index]
        
        data_dict = {
            "part_pcs": cur_pts,
            "gt_pcs": cur_pts_gt,
            "part_valids": valids,
            "part_quat": cur_quat,
            "part_trans": cur_trans,
            "n_pcs": n_pcs,
            "data_id": index,
            "critical_label_thresholds": label_thresholds,
            "mesh_file_path": mesh_file_path,
            "num_parts": num_parts,
        }
        return data_dict


def build_all_piece_matching_dataloader(cfg):
    data_dict = dict(
        data_dir=cfg.DATA.DATA_DIR,
        data_fn=cfg.DATA.DATA_FN.format("train"),
        data_keys=cfg.DATA.DATA_KEYS,
        category=cfg.DATA.CATEGORY,
        num_points=cfg.DATA.NUM_PC_POINTS,
        min_num_part=cfg.DATA.MIN_NUM_PART,
        max_num_part=cfg.DATA.MAX_NUM_PART,
        shuffle_parts=cfg.DATA.SHUFFLE_PARTS,
        rot_range=cfg.DATA.ROT_RANGE,
        overfit=cfg.DATA.OVERFIT,
        sample_by=cfg.DATA.SAMPLE_BY,
        min_part_point=cfg.DATA.MIN_PART_POINT,
        length=cfg.DATA.LENGTH * cfg.BATCH_SIZE,
        fracture_label_threshold=cfg.DATA.FRACTURE_LABEL_THRESHOLD,
    )
    train_set = AllPieceMatchingDataset(**data_dict)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    data_dict["data_fn"] = cfg.DATA.DATA_FN.format("val")
    data_dict["shuffle_parts"] = False
    data_dict["length"] = cfg.DATA.TEST_LENGTH

    val_set = AllPieceMatchingDataset(**data_dict)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=2,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    return train_loader, val_loader
