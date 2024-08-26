import os
import pickle

import numpy as np
import pytorch_lightning
import torch
from scipy.spatial.transform import Rotation as R
from torch import optim

from utils import Rotation3D, trans_metrics, rot_metrics, calc_part_acc, calc_shape_cd, global_alignment
from utils import filter_wd_parameters, CosineAnnealingWarmupRestarts
from utils import lexico_iter, get_trans_from_mat

import matplotlib.pyplot as plt
import time


class MatchingBaseModel(pytorch_lightning.LightningModule):
    def __init__(self, cfg):
        super(MatchingBaseModel, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self._setup()
        self.test_results = None
        self.cd_list = []
        if len(cfg.STATS):
            os.makedirs(cfg.STATS, exist_ok=True)
            self.stats = dict()
            self.stats['part_acc'] = []
            self.stats['chamfer_distance'] = []
            for metric in ['mse', 'rmse', 'mae']:
                self.stats[f'trans_{metric}'] = []
                self.stats[f'rot_{metric}'] = []
            self.stats['pred_trans'] = []
            self.stats['gt_trans'] = []
            self.stats['pred_rot'] = []
            self.stats['gt_rot'] = []
            self.stats['part_valids'] = []
        else:
            self.stats = None

    def _setup(self):
        self.max_num_part = self.cfg.DATA.MAX_NUM_PART

        self.pc_feat_dim = self.cfg.MODEL.PC_FEAT_DIM

    # The flow for this base model is:
    # training_step -> forward_pass -> loss_function ->
    # _loss_function -> forward

    def forward(self, data_dict):
        """Forward pass to predict matching."""
        raise NotImplementedError("forward function should be implemented per model")

    def training_step(self, data_dict, batch_idx, optimizer_idx=-1):
        loss_dict = self.forward_pass(
            data_dict, mode='train', optimizer_idx=optimizer_idx
        )
        return loss_dict['loss']

    def validation_step(self, data_dict, batch_idx):
        loss_dict = self.forward_pass(data_dict, mode='val', optimizer_idx=-1)
        return loss_dict

    def validation_epoch_end(self, outputs):
        # avg_loss among all data
        # we need to consider different batch_size

        func = torch.tensor if \
            isinstance(outputs[0]['batch_size'], int) else torch.stack
        batch_sizes = func([output.pop('batch_size') for output in outputs
                            ]).type_as(outputs[0]['loss'])  # [num_batches]
        losses = {
            f'val/{k}': torch.stack([output[k] for output in outputs]).reshape(-1)
            for k in outputs[0].keys()
        }  # each is [num_batches], stacked avg loss in each batch
        avg_loss = {
            k: (v * batch_sizes).sum() / batch_sizes.sum()
            for k, v in losses.items()
        }
        self.log_dict(avg_loss, sync_dist=True)

    def test_step(self, data_dict, batch_idx):
        torch.cuda.synchronize()
        start = time.time()
        loss_dict = self.forward_pass(data_dict, mode='test', optimizer_idx=-1)
        torch.cuda.synchronize()
        end = time.time()
        time_elapsed = end - start
        loss_dict['time'] = torch.tensor(time_elapsed, device=loss_dict['loss'].device, dtype=torch.float64)
        return loss_dict

    def test_epoch_end(self, outputs):
        # avg_loss among all data
        # we need to consider different batch_size
        if isinstance(outputs[0]['batch_size'], int):
            func_bs = torch.tensor
            func_loss = torch.stack
        else:
            func_bs = torch.cat
            func_loss = torch.cat
        batch_sizes = func_bs([output.pop('batch_size') for output in outputs
                               ]).type_as(outputs[0]['loss'])  # [num_batches]
        losses = {
            f'test/{k}': func_loss([output[k] for output in outputs])
            for k in outputs[0].keys()
        }  # each is [num_batches], stacked avg loss in each batch
        avg_loss = {
            k: (v * batch_sizes).sum() / batch_sizes.sum()
            for k, v in losses.items()
        }
        print('; '.join([f'{k}: {v.item():.6f}' for k, v in avg_loss.items()]))

        total_shape_cd = torch.mean(torch.cat(self.cd_list))
        print(f'total_shape_cd: {total_shape_cd.item():.6f}')

        # this is a hack to get results outside `Trainer.test()` function
        self.test_results = avg_loss
        self.log_dict(avg_loss, sync_dist=True)
        if self.cfg.STATS is not None:
            with open(os.path.join(self.cfg.STATS, 'saved_stats.pk'), 'wb') as f:
                pickle.dump(self.stats, f)

    @torch.no_grad()
    def _cus_vis(self, data_dict, pred_trans, pred_rot, gt_trans, gt_rot, part_acc):
        B = data_dict["num_parts"].shape[0]
        pred_trans_rots = torch.cat([pred_trans, pred_rot.to_quat()], dim=-1)
        gt_trans_tots = torch.cat([gt_trans, gt_rot.to_quat()], dim=-1)
        for i in range(B):
            save_dir = os.path.join("inference", str(data_dict["data_id"][i].item()))
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "mesh_file_path.txt"), "w") as f:
                f.write(data_dict["mesh_file_path"][i])
            mask = data_dict["part_valids"][i] == 1
            c_pred_trans_rots = pred_trans_rots[i, mask]
            c_gt_trans_rots = gt_trans_tots[i, mask]
            np.save(os.path.join(save_dir, f"predict_{part_acc[i]}.npy"), c_pred_trans_rots.cpu().numpy())
            np.save(os.path.join(save_dir, f"gt.npy"), c_gt_trans_rots.cpu().numpy())

            
        

    @torch.no_grad()
    def calc_metric(self, data_dict, trans_dict):
        """
        :param data_dict: must include:
            part_pcs: [B, P, 3]
            part_quat or part_rot: [B, P, 4] or [B, P, 3, 3], the ground truth quaternion or rotation
            part_trans: [B, P, 3], the ground truth transformation
            part_valids: [B, P], 1 for valid part, 0 for padding
        :param trans_dict: must include:
            rot: predicted rotation
            trans: predicted transformation
        :return: metric: will include
            6 eval metric, already the mean of the batch (total / (B*P_valid))
        """
        if 'part_rot' not in data_dict:
            part_quat = data_dict.pop('part_quat')
            data_dict['part_rot'] = \
                Rotation3D(part_quat, rot_type='quat').convert('rmat')
        part_valids = data_dict['part_valids']
        metric_dict = dict()
        part_pcs_nb = data_dict['part_pcs']
        pred_trans = torch.tensor(trans_dict['trans'], dtype=torch.float32, device=part_pcs_nb.device)
        pred_rot = torch.tensor(trans_dict['rot'], dtype=torch.float32, device=part_pcs_nb.device)
        pred_rot = Rotation3D(pred_rot, rot_type='rmat')
        gt_trans, gt_rot = data_dict['part_trans'], data_dict['part_rot']
        N_SUM = part_pcs_nb.shape[1]
        n_pcs = data_dict['n_pcs']
        B, P = n_pcs.shape
        part_pcs = []
        for b in range(B):
            point_sum = 0
            new_pcs = []
            for p in range(P):
                if n_pcs[b, p].item() == 0:
                    idx = torch.randint(low=point_sum - 1, high=point_sum, size=(N_SUM,))
                else:
                    idx = torch.randint(low=point_sum, high=point_sum + n_pcs[b, p].item(), size=(N_SUM,))
                new_pcs.append(part_pcs_nb[b, idx, :])
                point_sum += n_pcs[b, p]
            new_pcs = torch.stack(new_pcs)
            part_pcs.append(new_pcs)
        
        part_pcs = torch.stack(part_pcs).to(part_pcs_nb.device)
        part_acc, cd = calc_part_acc(part_pcs, pred_trans, gt_trans,
                                     pred_rot, gt_rot, part_valids, ret_cd=True)
        
        part_pcs_clone = part_pcs_nb.clone()
        shape_cd = calc_shape_cd(part_pcs_clone, n_pcs, pred_trans, pred_rot, data_dict["gt_pcs"], part_valids)
        
        self.cd_list.append(shape_cd)
        
        metric_dict['part_acc'] = part_acc.mean()
        metric_dict['chamfer_distance'] = shape_cd.mean()

        # self._cus_vis(data_dict, pred_trans, pred_rot, gt_trans, gt_rot, part_acc)

        for metric in ['mse', 'rmse', 'mae']:
            trans_met = trans_metrics(
                pred_trans, gt_trans, valids=part_valids, metric=metric)
            metric_dict[f'trans_{metric}'] = trans_met.mean()
            rot_met = rot_metrics(
                pred_rot, gt_rot, valids=part_valids, metric=metric)
            metric_dict[f'rot_{metric}'] = rot_met.mean()
            if self.stats is not None:
                self.stats[f'trans_{metric}'].append(trans_met.cpu().numpy())
                self.stats[f'rot_{metric}'].append(rot_met.cpu().numpy())
        if self.stats is not None:
            self.stats['part_acc'].append(part_acc.cpu().numpy())
            self.stats['chamfer_distance'].append(shape_cd.cpu().numpy())
            self.stats['pred_trans'].append(pred_trans.cpu().numpy())
            self.stats['gt_trans'].append(gt_trans.cpu().numpy())
            self.stats['pred_rot'].append(pred_rot.to_rmat().cpu().numpy())
            self.stats['gt_rot'].append(gt_rot.to_rmat().cpu().numpy())
            self.stats['part_valids'].append(part_valids.cpu().numpy())

        return metric_dict

    def _loss_function(self, data_dict, out_dict={}, optimizer_idx=-1):
        raise NotImplementedError("loss_function should be implemented per model")

    def transformation_loss(self, data_dict, out_dict):
        perm_mat = out_dict['perm_mat'].cpu().numpy()  # [B, N_, N_]
        ds_mat = out_dict['ds_mat'].cpu().numpy()  # [B, N_, N_]
        gt_pcs = data_dict['gt_pcs'].cpu().numpy()
        part_pcs = data_dict['part_pcs'].cpu().numpy()
        part_quat = data_dict['part_quat'].cpu().numpy()
        part_trans = data_dict['part_trans'].cpu().numpy()
        n_pcs = data_dict.get('n_pcs', None)
        if n_pcs is not None:
            n_pcs = n_pcs.cpu().numpy()

        part_valids = data_dict.get('part_valids', None)
        if part_valids is not None:
            part_valids = part_valids.cpu().numpy()
            n_valid = np.sum(part_valids, axis=1, dtype=np.int32)  # [B]
        else:
            n_valid = None

        gt_pcs = gt_pcs[:, :, :3]
        part_pcs = part_pcs[:, :, :3]
        B, N_sum, _ = gt_pcs.shape
        assert n_pcs is not None
        assert part_valids is not None
        assert n_valid is not None
        P = part_valids.shape[-1]
        N_ = ds_mat.shape[-1]

        critical_pcs_idx = data_dict.get('critical_pcs_idx', None)  # [B, N_sum]
        # critical_pcs_pos = data_dict.get('critical_pcs_pos', None)  # [\sum_B \sum_P n_critical_pcs[B, P], 3]
        n_critical_pcs = data_dict.get('n_critical_pcs', None)  # [B, P]

        n_critical_pcs = n_critical_pcs.cpu().numpy()
        critical_pcs_idx = critical_pcs_idx.cpu().numpy()

        density_match_mat = ds_mat
        best_match = np.argmax(ds_mat.reshape(-1, N_), axis=-1)
        density_match_mat_mask = np.zeros((B * N_, N_))
        density_match_mat_mask[np.arange(B * N_), best_match] = 1
        density_match_mat_mask = density_match_mat_mask.reshape(ds_mat.shape)

        pred_dict = self.compute_global_transformation(n_critical_pcs,
                                                       perm_mat,
                                                       gt_pcs, critical_pcs_idx,
                                                       part_pcs, n_valid, n_pcs,
                                                       part_quat, part_trans,
                                                       data_dict["data_id"],
                                                       data_dict["mesh_file_path"]
                                                       )
        metric_dict = self.calc_metric(data_dict, pred_dict)
        return metric_dict

    def compute_global_transformation(self, n_critical_pcs, match_mat, gt_pcs,
                                      critical_pcs_idx, part_pcs, n_valid, n_pcs,
                                      part_quat, part_trans, data_id, mesh_file_path):
        # B, P, N, _ = gt_pcs.shape
        B, N, _ = gt_pcs.shape
        P = n_critical_pcs.shape[-1]
        n_critical_pcs_cumsum = np.cumsum(n_critical_pcs, axis=-1)
        n_pcs_cumsum = np.cumsum(n_pcs, axis=-1)
        pred_dict = dict()
        pred_dict['rot'] = np.zeros((B, P, 3, 3))
        pred_dict['trans'] = np.zeros((B, P, 3))
        for b in range(B):
            if data_id[b].item() == 154:
                print("Debug")  
            
            piece_connections = np.zeros(n_valid[b])
            sum_full_matched = np.sum(match_mat[b])
            edges, transformations, uncertainty = [], [], []

            corr_list = []
            gt_pc_list = []
            critical_pcs_idx_list = []


            for idx1, idx2 in lexico_iter(np.arange(n_valid[b])): # All combinations of pieces
                cri_st1 = 0 if idx1 == 0 else n_critical_pcs_cumsum[b, idx1 - 1]
                cri_ed1 = n_critical_pcs_cumsum[b, idx1]
                cri_st2 = 0 if idx2 == 0 else n_critical_pcs_cumsum[b, idx2 - 1]
                cri_ed2 = n_critical_pcs_cumsum[b, idx2]

                pc_st1 = 0 if idx1 == 0 else n_pcs_cumsum[b, idx1 - 1]
                pc_ed1 = n_pcs_cumsum[b, idx1]
                pc_st2 = 0 if idx2 == 0 else n_pcs_cumsum[b, idx2 - 1]
                pc_ed2 = n_pcs_cumsum[b, idx2]

                np1 = n_pcs[b, idx1]
                np2 = n_pcs[b, idx2]
                n1 = n_critical_pcs[b, idx1]
                n2 = n_critical_pcs[b, idx2]
                if n1 == 0 or n2 == 0:
                    continue
                mat = match_mat[b, cri_st1:cri_ed1, cri_st2:cri_ed2]  # [N1, N2]
                mat_s = np.sum(mat).astype(np.int32)
                mat2 = match_mat[b, cri_st2:cri_ed2, cri_st1:cri_ed1]
                mat_s2 = np.sum(mat2).astype(np.int32)
                if mat_s < mat_s2:
                    mat = mat2.transpose(1, 0)
                    mat_s = mat_s2
                if n_valid[b] > 2 and mat_s == 0 and sum_full_matched > 0:
                    continue
                if np.count_nonzero(mat) < 3:
                    continue
                pc1 = part_pcs[b, pc_st1:pc_ed1]  # N, 3
                pc2 = part_pcs[b, pc_st2:pc_ed2]  # N, 3

                gt_pc1 = gt_pcs[b, pc_st1:pc_ed1]
                gt_pc2 = gt_pcs[b, pc_st2:pc_ed2]

                if critical_pcs_idx is not None:
                    critical_pcs_idx_1 = critical_pcs_idx[b, pc_st1: pc_st1 + n1]
                    critical_pcs_idx_2 = critical_pcs_idx[b, pc_st2: pc_st2 + n2]
                    
                    critical_pcs_src = pc1[critical_pcs_idx_1]
                    critical_pcs_tgt = pc2[critical_pcs_idx_2]
                    trans_mat, corr = get_trans_from_mat(critical_pcs_src, critical_pcs_tgt, mat)

                    # self._visualize_pair(
                    #     corr, 
                    #     gt_pc1, 
                    #     gt_pc2,
                    #     gt_pc1[critical_pcs_idx[b, pc_st1: pc_st1 + n1]],
                    #     gt_pc2[critical_pcs_idx[b, pc_st2: pc_st2 + n2]],
                    #     data_id[b].cpu().numpy().item(),
                    #     idx1,
                    #     idx2
                    # )

                    corr_list.append(corr)
                    gt_pc_list.append([gt_pc1, gt_pc2])
                    critical_pcs_idx_list.append([critical_pcs_idx_1, critical_pcs_idx_2])

                    edges.append(np.array([idx2, idx1]))
                    transformations.append(trans_mat)
                    uncertainty.append(1.0 / (mat_s))
                    piece_connections[idx1] = piece_connections[idx1] + 1
                    piece_connections[idx2] = piece_connections[idx2] + 1

            self._save_data(
                edges,
                corr_list,
                gt_pcs[b],
                critical_pcs_idx[b],
                n_pcs[b],
                n_critical_pcs[b],
                data_id[b].cpu().numpy().item(),
            )

                
            ## connect small pieces with less than 3 correspondence
            for idx1, idx2 in lexico_iter(np.arange(n_valid[b])):
                if piece_connections[idx1] > 0 and piece_connections[idx2] > 0:
                    continue
                if piece_connections[idx1] == 0 and piece_connections[idx2] == 0:
                    continue
                cri_st1 = 0 if idx1 == 0 else n_critical_pcs_cumsum[b, idx1 - 1]
                cri_ed1 = n_critical_pcs_cumsum[b, idx1]
                cri_st2 = 0 if idx2 == 0 else n_critical_pcs_cumsum[b, idx2 - 1]
                cri_ed2 = n_critical_pcs_cumsum[b, idx2]

                pc_st1 = 0 if idx1 == 0 else n_pcs_cumsum[b, idx1 - 1]
                pc_ed1 = n_pcs_cumsum[b, idx1]
                pc_st2 = 0 if idx2 == 0 else n_pcs_cumsum[b, idx2 - 1]
                pc_ed2 = n_pcs_cumsum[b, idx2]

                np1 = n_pcs[b, idx1]
                np2 = n_pcs[b, idx2]
                n1 = n_critical_pcs[b, idx1]
                n2 = n_critical_pcs[b, idx2]
                if n1 == 0 or n2 == 0:
                    edges.append(np.array([idx2, idx1]))
                    trans_mat = np.eye(4)
                    pc1 = part_pcs[b, pc_st1:pc_ed1]
                    pc2 = part_pcs[b, pc_st2:pc_ed2]
                    if n2 > 0:
                        trans_mat[:3, 3] = [critical_pcs_idx[b, pc_st2]] - np.sum(pc1, axis=0)
                    elif n1 > 0:
                        trans_mat[:3, 3] = np.sum(pc2, axis=0) - pc1[critical_pcs_idx[b, pc_st1]]
                    else:
                        trans_mat[:3, 3] = np.sum(pc2, axis=0) - np.sum(pc1, axis=0)
                    transformations.append(trans_mat)
                    uncertainty.append(1)
                    piece_connections[idx1] = piece_connections[idx1] + 1
                    piece_connections[idx2] = piece_connections[idx2] + 1
                    continue

                mat = match_mat[b, cri_st1:cri_ed1, cri_st2:cri_ed2]  # [N1, N2]
                mat_s = np.sum(mat).astype(np.int32)
                mat2 = match_mat[b, cri_st2:cri_ed2, cri_st1:cri_ed1]
                mat_s2 = np.sum(mat2).astype(np.int32)
                if mat_s < mat_s2:
                    mat = mat2.transpose(1, 0)
                    mat_s = mat_s2
                pc1 = part_pcs[b, pc_st1:pc_ed1]  # N, 3
                pc2 = part_pcs[b, pc_st2:pc_ed2]  # N, 3
                if critical_pcs_idx is not None:
                    critical_pcs_src = pc1[critical_pcs_idx[b, pc_st1: pc_st1 + n1]]
                    critical_pcs_tgt = pc2[critical_pcs_idx[b, pc_st2: pc_st2 + n2]]
                    trans_mat = np.eye(4)
                    matching1, matching2 = np.nonzero(mat)
                    trans_mat[:3, 3] = np.sum(critical_pcs_tgt[matching2], axis=0) - \
                                       np.sum(critical_pcs_src[matching1], axis=0)
                    edges.append(np.array([idx2, idx1]))
                    transformations.append(trans_mat)
                    uncertainty.append(1)
                    piece_connections[idx1] = piece_connections[idx1] + 1
                    piece_connections[idx2] = piece_connections[idx2] + 1
            # print(piece_connections, len(edges), len(uncertainty))
            if len(edges) > 0:
                edges = np.stack(edges)
                transformations = np.stack(transformations)
                uncertainty = np.array(uncertainty)
                global_transformations = global_alignment(n_valid[b], edges, transformations, uncertainty)
                pivot = 1
                for idx in range(n_valid[b]):
                    num_points = n_pcs[b, idx]
                    if num_points > n_pcs[b, pivot]:
                        pivot = idx
            else:
                global_transformations = np.repeat(np.eye(4).reshape((1, 4, 4)), n_valid[b], axis=0)
                pivot = 0
            to_gt_trans_mat = np.eye(4)
            quat = part_quat[b, pivot]
            to_gt_trans_mat[:3, :3] = R.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
            to_gt_trans_mat[:3, 3] = part_trans[b, pivot]

            offset = to_gt_trans_mat @ np.linalg.inv(global_transformations[pivot, :, :])
            for idx in range(n_valid[b]):
                global_transformations[idx, :, :] = offset @ global_transformations[idx, :, :]
            pred_dict['rot'][b, :n_valid[b], :, :] = global_transformations[:, :3, :3]
            pred_dict['trans'][b, :n_valid[b], :] = global_transformations[:, :3, 3]
        return pred_dict

    def loss_function(self, data_dict, optimizer_idx, mode):
        # loss_dict = None
        out_dict = self.forward(data_dict)

        loss_dict = self._loss_function(data_dict, out_dict, optimizer_idx)

        if 'loss' not in loss_dict:
            # if loss is composed of different losses, should combine them together
            # each part should be of shape [B, ] or [int]
            total_loss = 0.
            for k, v in loss_dict.items():
                if k.endswith('_loss'):
                    total_loss += v * eval(f'self.cfg.LOSS.{k.upper()}_W')
            loss_dict['loss'] = total_loss

        total_loss = loss_dict['loss']
        if total_loss.numel() != 1:
            loss_dict['loss'] = total_loss.mean()

        # log the batch_size for avg_loss computation
        if not self.training:
            if 'batch_size' not in loss_dict:
                loss_dict['batch_size'] = out_dict['batch_size']
        if mode == 'test':
            loss_dict.update(self.transformation_loss(data_dict, out_dict))
        return loss_dict

    def forward_pass(self, data_dict, mode, optimizer_idx):
        loss_dict = self.loss_function(data_dict, optimizer_idx=optimizer_idx, mode=mode)
        # in training we log for every step
        if mode == 'train' and self.local_rank == 0:
            log_dict = {f'{mode}/{k}': v.item() if isinstance(v, torch.Tensor) else v
                        for k, v in loss_dict.items()}
            data_name = [
                k for k in self.trainer.profiler.recorded_durations.keys()
                if 'prepare_data' in k
            ][0]
            log_dict[f'{mode}/data_time'] = \
                self.trainer.profiler.recorded_durations[data_name][-1]
            self.log_dict(
                log_dict, logger=True, sync_dist=False, rank_zero_only=True)
        return loss_dict

    def configure_optimizers(self):
        """Build optimizer and lr scheduler."""
        lr = self.cfg.TRAIN.LR
        wd = self.cfg.TRAIN.WEIGHT_DECAY

        if wd > 0.:
            params_dict = filter_wd_parameters(self)
            params_list = [{
                'params': params_dict['no_decay'],
                'weight_decay': 0.,
            }, {
                'params': params_dict['decay'],
                'weight_decay': wd,
            }]
            optimizer = optim.AdamW(params_list, lr=lr)
        else:
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.)

        if self.cfg.TRAIN.LR_SCHEDULER:
            assert self.cfg.TRAIN.LR_SCHEDULER.lower() in ['cosine']
            total_epochs = self.cfg.TRAIN.NUM_EPOCHS
            warmup_epochs = int(total_epochs * self.cfg.TRAIN.WARMUP_RATIO)
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                total_epochs,
                max_lr=lr,
                min_lr=lr / self.cfg.TRAIN.LR_DECAY,
                warmup_steps=warmup_epochs,
            )
            return (
                [optimizer],
                [{
                    'scheduler': scheduler,
                    'interval': 'epoch',
                }],
            )
        return optimizer


    # def _move_points(self, pc1, pc2, critical_pc1, critical_pc2, distance=0.1):
    #     # Move the points away from each other
    #     centroid1 = np.mean(pc1, axis=0)
    #     centroid2 = np.mean(pc2, axis=0)
        
    #     # Calculate the direction vector from centroid1 to centroid2
    #     direction = centroid2 - centroid1
    #     # Normalize the direction vector
    #     direction_norm = direction / np.linalg.norm(direction)
        
    #     # Move each point in pc1 away from centroid2
    #     pc1_moved = pc1 - direction_norm * distance
    #     # Move each point in pc2 away from centroid1
    #     pc2_moved = pc2 + direction_norm * distance

    #     # Move the critical points away from each other
    #     critical_pc1_moved = critical_pc1 - direction_norm * distance
    #     critical_pc2_moved = critical_pc2 + direction_norm * distance
        
    #     return pc1_moved, pc2_moved, critical_pc1_moved, critical_pc2_moved


    # def _visualize_pair(self, corr, gt_pc1, gt_pc2, critical_pc1, critical_pc2, data_id, idx1, idx2):
    #     # Visualize pair information
    #     save_dir = "vis_correspondance/vis_by_data_id" 
    #     save_dir = os.path.join(save_dir, f"{data_id}")
    #     os.makedirs(save_dir, exist_ok=True)
    #     save_path = os.path.join(save_dir, f"{idx1}_{idx2}.png")
        
    #     # TODO: 1. Visualize teh ground truth

    #     # TODO: 2. Visualize the correspondence for the points
    #     # 2.1: Move all points away from each other
    #     gt_pc1_moved, gt_pc2_moved, critical_pc1_moved, critical_pc2_moved = self._move_points(gt_pc1, gt_pc2, critical_pc1, critical_pc2)
        
    #     # 2.1: Get the correspondence points
    #     corr_points_1 = critical_pc1_moved[corr[:, 0]]
    #     corr_points_2 = critical_pc2_moved[corr[:, 1]]
        
    #     # 2.3: Visualize the correspondence points by connecting them with lines
    #     # 2.3.1: Create a new plot
    #     fig = plt.figure(figsize=(14, 7))
    #     ax1 = fig.add_subplot(121, projection='3d')
    #     ax1.set_xlim([- 0.5, 0.5])
    #     ax1.set_ylim([- 0.5, 0.5])
    #     ax1.set_zlim([- 0.5, 0.5])

    #     # 2.3.2: Plot the points
    #     ax1.scatter(gt_pc1_moved[:, 0], gt_pc1_moved[:, 1], gt_pc1_moved[:, 2], c='r', s=1)
    #     ax1.scatter(gt_pc2_moved[:, 0], gt_pc2_moved[:, 1], gt_pc2_moved[:, 2], c='b', s=1)
    #     ax1.scatter(corr_points_1[:, 0], corr_points_1[:, 1], corr_points_1[:, 2], c='g')
    #     ax1.scatter(corr_points_2[:, 0], corr_points_2[:, 1], corr_points_2[:, 2], c='g')

    #     # 2.3.3: Connect the correspondence points with lines
    #     for i in range(corr_points_1.shape[0]):
    #         ax1.plot(
    #             [corr_points_1[i, 0], corr_points_2[i, 0]], 
    #             [corr_points_1[i, 1], corr_points_2[i, 1]], 
    #             [corr_points_1[i, 2], corr_points_2[i, 2]], 
    #             c='g', linewidth=0.1
    #         )

    #     ax2 = fig.add_subplot(122, projection='3d')
    #     # Visualize ground truth points on the right side
    #     ax2.scatter(gt_pc1[:, 0], gt_pc1[:, 1], gt_pc1[:, 2], c='r', s=1)
    #     ax2.scatter(gt_pc2[:, 0], gt_pc2[:, 1], gt_pc2[:, 2], c='b', s=1)
        
    #     ax2.set_xlim([- 0.5, 0.5])
    #     ax2.set_ylim([- 0.5, 0.5])
    #     ax2.set_zlim([- 0.5, 0.5])
    #     # 2.3.5: Show the plot
    #     plt.savefig(save_path)
    #     plt.close()



    def _save_data(self, edges, corr_list, gt_pcs, critical_pcs_idx, n_pcs, n_critical_pcs, data_id):
        """
        save to a dictionary 
        edges: current 2 index of the pair
        correspondance: list of correspondance of critical points
        gt_pc: list of ground truth points
        critical_pcs: list of critical points
        """
        save_dir = "matching_data/everyday"
        os.makedirs(save_dir, exist_ok=True)

        # save to npz
        save_path = os.path.join(save_dir, f"{data_id}.npz")
        # if save_path exists, skip
        if os.path.exists(save_path):
            return
        
        data = {
            "edges": edges,
            "correspondence": corr_list,
            "gt_pcs": gt_pcs,
            "critical_pcs_idx": critical_pcs_idx,
            "n_pcs": n_pcs,
            "n_critical_pcs": n_critical_pcs
        }

        np.savez(save_path, **data)
        

