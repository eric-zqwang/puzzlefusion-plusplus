"""
Code to generate point cloud data from the dataset.
"""

import hydra
from puzzlefusion_plusplus.vqvae.dataset.dataset import build_geometry_dataloader
import os
import numpy as np
from tqdm import tqdm

@hydra.main(config_path='config/ae', config_name='global_config.yaml')
def main(cfg):
    cfg.data.batch_size = 1
    cfg.data.val_batch_size = 1
    train_loader, val_loader = build_geometry_dataloader(cfg)
    
    def save_data(loader, data_type):
        save_path = f"{cfg.data.save_pc_data_path}/{data_type}/"
        os.makedirs(save_path, exist_ok=True)

        for i, data_dict in tqdm(enumerate(loader), total=len(loader), desc=f"Processing {data_type} data"):
            data_id = data_dict['data_id'][0].item()
            part_valids = data_dict['part_valids'][0]
            num_parts = data_dict['num_parts'][0].item()
            mesh_file_path = data_dict['mesh_file_path'][0]
            graph = data_dict['graph'][0]
            category = data_dict['category'][0]
            part_pcs_gt = data_dict['part_pcs_gt'][0]
            ref_part = data_dict['ref_part'][0]

            np.savez(
                os.path.join(save_path, f'{data_id:05}.npz'),
                data_id=data_id,
                part_valids=part_valids.cpu().numpy(),
                num_parts=num_parts,
                mesh_file_path=mesh_file_path,
                graph=graph.cpu().numpy(),
                category=category,
                part_pcs_gt=part_pcs_gt.cpu().numpy(),
                ref_part=ref_part.cpu().numpy()
            )
            # print(f"Saved {data_id:05}.npz in {data_type} data.")

    # Save train data
    save_data(train_loader, 'train')
    # Save validation data
    save_data(val_loader, 'val')

# python generate_pc_data.py +data.save_pc_data_path=pc_data/everyday
if __name__ == '__main__':
    main()