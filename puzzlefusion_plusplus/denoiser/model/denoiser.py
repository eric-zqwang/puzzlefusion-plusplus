import torch
from torch.nn import functional as F
import lightning.pytorch as pl
import hydra
from puzzlefusion_plusplus.denoiser.model.modules.denoiser_transformer import DenoiserTransformer
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


class Denoiser(pl.LightningModule):
    def __init__(self, cfg):
        super(Denoiser, self).__init__()
        self.cfg = cfg
        self.denoiser = DenoiserTransformer(cfg)

        self.save_hyperparameters()

        self.noise_scheduler = PiecewiseScheduler(
            num_train_timesteps=cfg.model.DDPM_TRAIN_STEPS,
            beta_schedule=cfg.model.DDPM_BETA_SCHEDULE,
            prediction_type=cfg.model.PREDICT_TYPE,
            beta_start=cfg.model.BETA_START,
            beta_end=cfg.model.BETA_END,
            clip_sample=False,
            timestep_spacing=self.cfg.model.timestep_spacing
        )

        self.encoder = hydra.utils.instantiate(cfg.ae.ae_name, cfg)

        self.cd_loss = ChamferDistance()
        self.num_points = cfg.model.num_point
        self.num_channels = cfg.model.num_dim

        self.noise_scheduler.set_timesteps(
            num_inference_steps=cfg.model.num_inference_steps
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


    def forward(self, data_dict):
        gt_trans = data_dict['part_trans']
        gt_rots = data_dict['part_rots']
        gt_trans_and_rots = torch.cat([gt_trans, gt_rots], dim=-1)
        ref_part = data_dict["ref_part"]
        noise = torch.randn(gt_trans_and_rots.shape, device=self.device)

        B, P, N, C = data_dict["part_pcs"].shape

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,),
                                  device=self.device).long()
        
        noisy_trans_and_rots = self.noise_scheduler.add_noise(gt_trans_and_rots, noise, timesteps)

        noisy_trans_and_rots[ref_part] = gt_trans_and_rots[ref_part]

        part_pcs = data_dict["part_pcs"]
        part_valids = data_dict["part_valids"]
        latent, xyz = self._extract_features(part_pcs, part_valids, noisy_trans_and_rots)

        pred_noise = self.denoiser(
            noisy_trans_and_rots, 
            timesteps, 
            latent, 
            xyz, 
            data_dict['part_valids'],
            data_dict['part_scale'],
            ref_part
        )

        output_dict = {
            'pred_noise': pred_noise,
            'gt_noise': noise
        }

        return output_dict


    def _loss(self, data_dict, output_dict):
        pred_noise = output_dict['pred_noise']
        part_valids = data_dict['part_valids'].bool()
        noise = output_dict['gt_noise']

        part_valids[data_dict["ref_part"]] = False
        mse_loss = F.mse_loss(pred_noise[part_valids], noise[part_valids])

        return {'mse_loss': mse_loss}


    def training_step(self, data_dict, idx):
        output_dict = self(data_dict)
        loss_dict = self._loss(data_dict, output_dict)
        
        total_loss = 0
        for loss_name, loss_value in loss_dict.items():
            total_loss += loss_value
            self.log(f"train_loss/{loss_name}", loss_value, on_step=True, on_epoch=False)
        self.log(f"train_loss/total_loss", total_loss, on_step=True, on_epoch=False)
        
        return total_loss
    

    def _calc_val_loss(self, data_dict):
        output_dict = self(data_dict)
        loss_dict = self._loss(data_dict, output_dict)
        # calculate the total loss and logs
        total_loss = 0
        for loss_name, loss_value in loss_dict.items():
            total_loss += loss_value
            self.log(f"val_loss/{loss_name}", loss_value, on_step=False, on_epoch=True)
        self.log(f"val_loss/total_loss", total_loss, on_step=False, on_epoch=True)
                

    def validation_step(self, data_dict, idx):
        self._calc_val_loss(data_dict)
        
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


        pts = data_dict['part_pcs']
        pred_trans = noisy_trans_and_rots[..., :3]
        pred_rots = noisy_trans_and_rots[..., 3:]

        expanded_part_scale = data_dict["part_scale"].unsqueeze(-1).expand(-1, -1, 1000, -1)
        pts = pts * expanded_part_scale

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


    def on_validation_epoch_end(self):
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
