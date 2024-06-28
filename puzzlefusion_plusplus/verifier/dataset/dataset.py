import os
import numpy as np
from scipy.spatial.transform import Rotation as R

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import copy


class VerifierDataset(Dataset):
    """
    Datset for training the discriminator.
    
    data:
        - point cloud for sample by area
        - matching points
        - transformed parameters

    Processed data:
        - matching points distance; use a vector to represent
            e.g. [90, 20, 14, 0, 0, 0] -> Each number represent the number of points in the distance range
            Use a PE to make it to higher dimension; 6 x 1 -> 6 x D
    """

    def __init__(
            self,
            data_dir,
            overfit,
            mode
    ):
        self.max_nodes = 20
        self.max_edges = int(self.max_nodes * (self.max_nodes - 1) / 2)

        self.data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        self.data_list = []
        
        if overfit != -1:
            self.data_files = self.data_files[:overfit]

        # if mode = 'train' first 80% of the data is used for training
        # if mode = 'val' last 20% of the data is used for validation
        if mode == 'train':
            self.data_files = self.data_files[:int(0.8 * len(self.data_files))]
        elif mode == 'val':
            val_offset = int(0.8 * len(self.data_files))
            self.data_files = self.data_files[val_offset:]


        for file in tqdm(self.data_files):
            data = np.load(os.path.join(data_dir, file))
            cls_gt = data["cls_gt"].astype(np.int64)
            edge_features = data["edge_features"]
            edge_indices = data["edge_indices"]
            num_edges = edge_indices.shape[0]

            cls_gt = self._pad_data(cls_gt).astype(np.float32)
            edge_features = self._pad_data(edge_features).astype(np.float32)
            edge_indices = self._pad_data(edge_indices).astype(np.int64)

            edge_valids = np.zeros(self.max_edges, dtype=np.float32)
            edge_valids[:num_edges] = 1

            data_dict = {
                "cls_gt": cls_gt,
                "edge_features": edge_features,
                "edge_indices": edge_indices,
                "edge_valids": edge_valids,
                "num_edges": num_edges,
            }

            self.data_list.append(data_dict)

        print("Finished loading data")
        
    
    def _pad_data(self, data):
        max_edges = self.max_edges
        pad_shape = (max_edges, ) + tuple(data.shape[1:])
        pad_data = np.zeros(pad_shape, dtype=np.float32)
        pad_data[:data.shape[0]] = data
        return pad_data
    

    def __len__(self):
        return len(self.data_list) 


    def __getitem__(self, index):
        data_dict = copy.deepcopy(self.data_list[index])
        edge_features = data_dict["edge_features"]

        # Normalize the edge features based on min and max values
        num_points = np.sum(edge_features, axis=1)
        edge_features = edge_features / np.where(num_points == 0, 1, num_points)[:, np.newaxis]
        
        # Append number of matched points
        edge_features = np.concatenate([edge_features, num_points[:, np.newaxis]], axis=1)
        data_dict["edge_features"] = edge_features
        
        return data_dict
    

def build_geometry_dataloader(cfg):

    data_dict = dict(
        data_dir=cfg.data.verifier_data_path,
        overfit=cfg.data.overfit,
        mode="train"
    )
    train_set = VerifierDataset(**data_dict)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.data.num_workers > 0),
    )
    
    data_dict['mode'] = "val"
    val_set = VerifierDataset(**data_dict)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.data.val_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.data.num_workers > 0),
    )

    return train_loader, val_loader
