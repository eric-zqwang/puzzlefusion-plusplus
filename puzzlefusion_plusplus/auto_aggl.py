import torch
from torch.nn import functional as F
import lightning.pytorch as pl
import hydra
from puzzlefusion_plusplus.denoiser.model.modules.denoiser_transformer import DenoiserTransformer
from puzzlefusion_plusplus.verifier.model.modules.verifier_transformer import VerifierTransformer
from puzzlefusion_plusplus.vqvae.model.modules.vq_vae import VQVAE
from tqdm import tqdm
from chamferdist import ChamferDistance
from puzzlefusion_plusplus.denoiser.evaluation.evaluator import (
    calc_part_acc,
    trans_metrics,
    rot_metrics,
    calc_shape_cd
)
import numpy as np
from puzzlefusion_plusplus.denoiser.model.modules.custom_diffusers import PiecewiseScheduler
from pytorch3d import transforms
import networkx as nx    
import itertools
import os
from utils.node_merge_utils import (
    get_final_pose_pts_dynamic,
    get_final_pose_pts,
    get_distance_for_matching_pts,
    node_merge_valids_check,
    get_pc_start_end,
    merge_node,
    get_param,
    assign_init_pose,
    remove_intersect_points_and_fps_ds,
    extract_final_pred_trans_rots
)

class AutoAgglomerative(pl.LightningModule):
    def __init__(self, cfg):
        super(AutoAgglomerative, self).__init__()
        self.cfg = cfg
        self.denoiser = DenoiserTransformer(cfg.denoiser)
        self.verifier = VerifierTransformer(cfg.verifier)
        self.encoder = VQVAE(cfg.ae)
        
        self.save_hyperparameters()

        self.noise_scheduler = PiecewiseScheduler(
            num_train_timesteps=cfg.denoiser.model.DDPM_TRAIN_STEPS,
            beta_schedule=cfg.denoiser.model.DDPM_BETA_SCHEDULE,
            prediction_type=cfg.denoiser.model.PREDICT_TYPE,
            beta_start=cfg.denoiser.model.BETA_START,
            beta_end=cfg.denoiser.model.BETA_END,
            clip_sample=False,
            timestep_spacing=self.cfg.denoiser.model.timestep_spacing
        )

        self.num_points = cfg.denoiser.model.num_point
        self.num_channels = cfg.denoiser.model.num_dim

        self.noise_scheduler.set_timesteps(
            num_inference_steps=cfg.denoiser.model.num_inference_steps
        )

        self.rmse_r_list = []
        self.rmse_t_list = []
        self.acc_list = []
        self.cd_list = []

        self.metric = ChamferDistance()


    def _apply_rots(self, part_pcs, noise_params):
        """
        Apply Noisy rotations to all points
        """
        noise_quat = noise_params[..., 3:]
        noise_quat = noise_quat / noise_quat.norm(dim=-1, keepdim=True)
        part_pcs = transforms.quaternion_apply(noise_quat.unsqueeze(2), part_pcs)
        
        return part_pcs
    

    def _extract_features(self, part_pcs, part_valids, noisy_trans_and_rots):
        B, P , _, _ = part_pcs.shape
        part_pcs = self._apply_rots(part_pcs, noisy_trans_and_rots)
        part_pcs = part_pcs[part_valids.bool()]

        encoder_out = self.encoder.encode(part_pcs)
        latent = torch.zeros(B, P, self.num_points, self.num_channels, device=self.device)
        xyz = torch.zeros(B, P, self.num_points, 3, device=self.device)

        latent[part_valids.bool()] = encoder_out["z_q"]
        xyz[part_valids.bool()] = encoder_out["xyz"]
        return latent, xyz


    def test_step(self, data_dict, idx):        
        gt_trans = data_dict['part_trans']
        gt_rots = data_dict['part_rots']
        gt_trans_and_rots = torch.cat([gt_trans, gt_rots], dim=-1)
        noisy_trans_and_rots = torch.randn(gt_trans_and_rots.shape, device=self.device)
        ref_part = data_dict["ref_part"]        

        reference_gt_and_rots = torch.zeros_like(gt_trans_and_rots, device=self.device)
        reference_gt_and_rots[ref_part] = gt_trans_and_rots[ref_part]
        noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part]

        part_valids = data_dict['part_valids'].clone()
        part_scale = data_dict["part_scale"].clone()
        part_pcs = data_dict["part_pcs"].clone()
        B, P, N, _  = part_pcs.shape

        num_parts = data_dict["num_parts"].clone()

        # pre-computed edges, corresponding index, dynamic sample points, surface points
        edges = data_dict['edges']
        corr = data_dict['correspondences']
        part_pcs_by_area = data_dict['part_pcs_by_area']
        critical_pcs_idx = data_dict['critical_pcs_idx']
        n_pcs = data_dict['n_pcs']
        n_critical_pcs = data_dict['n_critical_pcs']

        # initialize the graph
        G = nx.Graph()
        for i in range(num_parts[0].item()):
            G.add_node(
                i, 
                pivot=i, 
                valids=True, 
                ref_part=False,
                init_pose=None,
            )
        G.nodes[torch.where(ref_part)[1].item()]["ref_part"] = True
        classified_part = torch.zeros_like(data_dict['part_valids'], device=self.device).to(torch.bool)
        
        all_pred_trans_rots = []

        for iter in range(self.cfg.verifier.max_iters):
            for t in self.noise_scheduler.timesteps:
                timesteps = t.reshape(-1).repeat(len(noisy_trans_and_rots)).cuda()
                latent, xyz = self._extract_features(part_pcs, part_valids, noisy_trans_and_rots)
                pred_noise = self.denoiser(
                    noisy_trans_and_rots, 
                    timesteps,
                    latent,
                    xyz,
                    part_valids,
                    part_scale,
                    ref_part
                )
                noisy_trans_and_rots = self.noise_scheduler.step(pred_noise, t, noisy_trans_and_rots).prev_sample
                noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part]  
                all_pred_trans_rots.append(get_param(noisy_trans_and_rots[0], G.nodes).unsqueeze(0).cpu().numpy())

            if iter + 1 == self.cfg.verifier.max_iters:
                break

            pts = part_pcs.clone()
            pred_trans = noisy_trans_and_rots[..., :3]
            pred_rots = noisy_trans_and_rots[..., 3:]

            expanded_part_scale = part_scale.unsqueeze(-1).expand(-1, -1, 1000, -1)
            pts = pts * expanded_part_scale
            transformed_pts = get_final_pose_pts(pts, pred_trans, pred_rots)

            part_pcs_by_area_transformed = get_final_pose_pts_dynamic(
                part_pcs_by_area,
                n_pcs,
                pred_trans,
                pred_rots,
                num_parts,
                G.nodes
            )

            part_valids_bool = part_valids.to(torch.bool)

            ref_part_idx = torch.where(ref_part)[1]
            classified_part[0, ref_part_idx] = True

            larger_parts = part_valids_bool & (part_scale.squeeze(2) > 0.05)

            new_ref_part_idx_list = []
            edge_features = torch.zeros(B, P, P, 6, dtype=torch.int32, device=self.device)
            
            # For reference part
            for i in range(edges.shape[1]):
                idx2 = edges[0, i, 0]
                idx1 = edges[0, i, 1]
                cd_per_point = get_distance_for_matching_pts(
                    idx1, idx2, part_pcs_by_area_transformed,n_pcs, 
                    n_critical_pcs,critical_pcs_idx, corr[i],
                    data_dict["data_id"][0].item(), self.metric
                )
                bins = self._make_cd_to_bins(cd_per_point)
                edge_features[0, idx1, idx2] = bins
            
            mat_mask = torch.triu(torch.ones(P, P, dtype=torch.bool, device=self.device), diagonal=1)
            edge_features = edge_features[:, mat_mask]
            edge_valids = self._get_edge_mask(num_parts, P)
            edge_indices = mat_mask.nonzero(as_tuple=False).unsqueeze(0)
            num_points = edge_features.sum(dim=-1, keepdim=True)
            edge_features = edge_features / torch.where(num_points == 0, 1, num_points)
            edge_features = torch.cat((edge_features, num_points), dim=-1)

            logits = self.verifier(edge_features, edge_indices, edge_valids)
            scores = torch.sigmoid(logits)
            pred_labels = (scores > self.cfg.verifier.threshold).squeeze(-1) & edge_valids
            classified_edges = edge_indices[pred_labels]

            for edge in classified_edges:
                idx1, idx2 = edge
                if (part_valids_bool[0][idx1] or part_valids_bool[0][idx2]) is False:
                    continue 
                if idx1 in ref_part_idx and idx2 in ref_part_idx: # both are reference part
                    continue
                if idx1 not in ref_part_idx and idx2 not in ref_part_idx: # both are non reference part
                    continue
                non_ref_part_idx = idx1 if idx1 not in ref_part_idx else idx2
                new_ref_part_idx_list.append(non_ref_part_idx) 

            for non_ref_part_idx in new_ref_part_idx_list:
                ref_part[0][non_ref_part_idx] = True

            reference_gt_and_rots = noisy_trans_and_rots.clone()

            node_merge_list = []
            for edge in classified_edges:
                idx1, idx2 = edge
                if node_merge_valids_check(edge, ref_part, G.nodes):
                    node_merge_list.append((idx1.item(), idx2.item()))

            # if all parts classified. then stop
            if (classified_part == larger_parts).all():
                break

            if len(node_merge_list) > 0:
                G.add_edges_from(node_merge_list)

                connected_component = list(nx.connected_components(G))
                for component in connected_component:
                    component = list(component)

                    num_valids = 0
                    for c in component:
                        num_valids += G.nodes[c]['valids']

                    if num_valids <= 1: # do not have new node to merge
                        continue

                    # Select the part scale largest as pivot
                    pivot = max(component, key=lambda x: part_scale[0][x])
                    merge_pcs = merge_node(component, G.nodes, transformed_pts[0])

                    # recenter the part
                    centroid = merge_pcs.mean(dim=0)
                    merge_pcs = merge_pcs - centroid

                    assign_init_pose(G.nodes, pred_trans[0], pred_rots[0], centroid, component)

                    # Update trans bias
                    for c in component:
                        pc_st1, pc_ed1 = get_pc_start_end(c, n_pcs)
                        final_pose_by_area = part_pcs_by_area_transformed[pc_st1:pc_ed1] - centroid
                        part_pcs_by_area[0, pc_st1:pc_ed1] = final_pose_by_area

                    
                    for c in component:
                        G.nodes[c]['pivot'] = pivot
                    
                    merge_pcs_ds = remove_intersect_points_and_fps_ds(merge_pcs, self.metric)

                    # normalize pc to [-1, 1]
                    merge_scale = merge_pcs_ds.abs().max()
                    merge_pcs_ds = merge_pcs_ds / merge_scale
                    # update the part scale and pcs
                    part_scale[0][pivot] = merge_scale
                    part_pcs[0][pivot] = merge_pcs_ds
                    # update the part valids
                    part_valids[0, component] = 0
                    part_valids[0, pivot] = 1

                    for c in component:
                        if c == pivot:
                            G.nodes[c]['valids'] = True
                        else:
                            G.nodes[c]['valids'] = False
                    classified_part[0, component] = True
                node_merge_list = []

            if (classified_part == larger_parts).all():
                break

        pts = data_dict['part_pcs']
        pred_trans = noisy_trans_and_rots[..., :3]
        pred_rots = noisy_trans_and_rots[..., 3:]

        expanded_part_scale = data_dict["part_scale"].unsqueeze(-1).expand(-1, -1, 1000, -1)
        pts = pts * expanded_part_scale

        pred_trans, pred_rots = extract_final_pred_trans_rots(pred_trans[0], pred_rots[0], G.nodes)

        pred_trans = pred_trans.unsqueeze(0)
        pred_rots = pred_rots.unsqueeze(0)

        acc, _, _ = calc_part_acc(pts, trans1=pred_trans, trans2=gt_trans,
                            rot1=pred_rots, rot2=gt_rots, valids=data_dict['part_valids'], 
                            chamfer_distance=self.metric)
        
        shape_cd = calc_shape_cd(pts, trans1=pred_trans, trans2=gt_trans,
                            rot1=pred_rots, rot2=gt_rots, valids=data_dict['part_valids'], 
                            chamfer_distance=self.metric)
        
        rmse_r = rot_metrics(pred_rots, gt_rots, data_dict['part_valids'], 'rmse')
        rmse_t = trans_metrics(pred_trans, gt_trans,  data_dict['part_valids'], 'rmse')
        
        self.acc_list.append(acc)
        self.rmse_r_list.append(rmse_r)
        self.rmse_t_list.append(rmse_t)
        self.cd_list.append(shape_cd)

        self._save_inference_data(data_dict, np.stack(all_pred_trans_rots, axis=0), acc)


    def _save_inference_data(self, data_dict, pred_trans_rots, acc):
        T, B, _, _ = pred_trans_rots.shape

        for i in range(B):
            save_dir = os.path.join(
                self.cfg.experiment_output_path,
                "inference", 
                self.cfg.inference_dir, 
                str(data_dict['data_id'][i].item())
            )

            os.makedirs(save_dir, exist_ok=True)
            c_trans_rots = pred_trans_rots[:, i, ...]
            mask = data_dict["part_valids"][i] == 1
            c_trans_rots = c_trans_rots[:, mask.cpu().numpy(), ...]
            np.save(os.path.join(save_dir, f"predict_{acc[i]}.npy"), c_trans_rots)
            gt_transformation = torch.cat(
                [data_dict["part_trans"][i],
                    data_dict["part_rots"][i]], dim=-1
            )[mask]

            np.save(os.path.join(
                save_dir, "gt.npy"),
                gt_transformation.cpu().numpy()
            )

            init_pose_r = data_dict["init_pose_r"][i]
            init_pose_t = data_dict["init_pose_t"][i]
            init_pose = torch.cat([init_pose_t, init_pose_r], dim=-1)
            np.save(os.path.join(
                save_dir, "init_pose.npy"),
                init_pose.cpu().numpy()
            )

            with open(os.path.join(save_dir, "mesh_file_path.txt"), "w") as f:
                f.write(data_dict["mesh_file_path"][i])


    def on_test_epoch_end(self):
        total_acc = torch.mean(torch.cat(self.acc_list))
        total_rmse_t = torch.mean(torch.cat(self.rmse_t_list))
        total_rmse_r = torch.mean(torch.cat(self.rmse_r_list))
        total_shape_cd = torch.mean(torch.cat(self.cd_list))
        
        self.log(f"eval/part_acc", total_acc, sync_dist=True)
        self.log(f"eval/rmse_t", total_rmse_t, sync_dist=True)
        self.log(f"eval/rmse_r", total_rmse_r, sync_dist=True)
        self.log(f"eval/shape_cd", total_shape_cd, sync_dist=True)
        self.acc_list = []
        self.rmse_t_list = []
        self.rmse_r_list = []
        self.cd_list = []
        return total_acc, total_rmse_t, total_rmse_r, total_shape_cd

    def _get_edge_mask(self, num_parts, P):
        B = num_parts.shape[0]
        nodes = range(P)
        edges = list(itertools.combinations(nodes, 2))
        edges = torch.tensor(edges, dtype=torch.int32, device=self.device)
        edges = edges.unsqueeze(0).expand(B, -1, -1)
        mask = (edges[:, :, 0] < num_parts.unsqueeze(1)) & (edges[:, :, 1] < num_parts.unsqueeze(1))
        return mask
    
    def _make_cd_to_bins(self, cd):
        bins = torch.tensor([0.0, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 100], device=self.device)
        bin_indices = torch.bucketize(cd.squeeze(0), bins, right=True)
        counts = torch.bincount(bin_indices, minlength=bins.numel())
        return counts[1:7]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=2e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )
        lr_scheduler = hydra.utils.instantiate(self.cfg.model.lr_scheduler, optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
