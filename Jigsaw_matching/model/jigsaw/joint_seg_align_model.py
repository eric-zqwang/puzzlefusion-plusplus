import torch
import torch.nn.functional as fun
import torchmetrics
from torch import nn

from model import MatchingBaseModel, build_encoder
from utils import Sinkhorn, hungarian
from utils import get_batch_length_from_part_points, square_distance
from utils import get_critical_pcs_from_label
from utils import permutation_loss, rigid_loss
from .affinity_layer import build_affinity
from .attention_layer import PointTransformerLayer, CrossAttentionLayer
import numpy as np
import os

class JointSegmentationAlignmentModel(MatchingBaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.num_classes = 2
        self.aff_feat_dim = self.cfg.MODEL.AFF_FEAT_DIM
        assert self.aff_feat_dim % 2 == 0, "The affinity feature dimension must be even!"
        self.half_aff_feat_dim = self.aff_feat_dim // 2

        self.w_cls_loss = self.cfg.MODEL.LOSS.w_cls_loss
        self.w_mat_loss = self.cfg.MODEL.LOSS.w_mat_loss
        self.w_rig_loss = self.cfg.MODEL.LOSS.w_rig_loss
        print(
            f"initial: self.w_cls_loss {self.w_cls_loss}, self.w_mat_loss"
            f" {self.w_mat_loss}, self.w_rig_loss {self.w_rig_loss}"
        )

        self.pc_cls_method = self.cfg.MODEL.PC_CLS_METHOD.lower()
        print(f"self.pc_cls_method: {self.pc_cls_method}")
        # options: ['binary', 'multi'].
        # We also provide a method which treats the segmentation as a multi-class classification problem.
        self.num_classes = self.cfg.MODEL.PC_NUM_CLS
        self.encoder = self._init_encoder()
        self.affinity_layer = self._init_affinity_layer()
        self.sinkhorn = self._init_sinkhorn()
        self.pc_classifier = self._init_classifier()
        self.affinity_extractor = self._init_affinity_extractor()

        self.tf_self1 = PointTransformerLayer(
            in_feat=self.pc_feat_dim, out_feat=self.pc_feat_dim,
            n_heads=self.cfg.MODEL.TF_NUM_HEADS, nsampmle=self.cfg.MODEL.TF_NUM_SAMPLE,
        )
        self.tf_cross1 = CrossAttentionLayer(d_in=self.pc_feat_dim,
                                             n_head=self.cfg.MODEL.TF_NUM_HEADS,)
        self.tf_layers = [("self", self.tf_self1), ("cross", self.tf_cross1)]

        if not self.cfg.MODEL.TEST_S_MASK:
            # default: True. The mask is not needed based on the design of the primal-dual descriptor.
            print("No mask for s in test.")

    def _init_encoder(self):
        in_feat_dim = 3
        encoder = build_encoder(
            self.cfg.MODEL.ENCODER,
            feat_dim=self.pc_feat_dim,
            global_feat=False,
            in_feat_dim=in_feat_dim,
        )
        return encoder

    def _init_affinity_extractor(self):
        affinity_extractor = nn.Sequential(
            nn.BatchNorm1d(self.pc_feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.pc_feat_dim, self.aff_feat_dim, 1),
        )
        return affinity_extractor

    def _init_classifier(self):
        if self.pc_cls_method == 'binary':
            classifier = nn.Sequential(
                nn.BatchNorm1d(self.pc_feat_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.pc_feat_dim, 1, 1),
            )
        elif self.pc_cls_method == 'multi':
            classifier = nn.Sequential(
                nn.BatchNorm1d(self.pc_feat_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.pc_feat_dim, self.num_classes, 1)
            )
        else:
            raise NotImplementedError(f"{self.pc_cls_method} not implemented for classifier")
        return classifier

    def _init_affinity_layer(self):
        affinity_layer = build_affinity(
            self.cfg.MODEL.AFFINITY.lower(), self.aff_feat_dim
        )
        return affinity_layer

    def _init_sinkhorn(self):
        return Sinkhorn(
            max_iter=self.cfg.MODEL.SINKHORN_MAXITER, tau=self.cfg.MODEL.SINKHORN_TAU
        )

    def _extract_part_feats(self, part_pcs, batch_length):
        B, N_sum, _ = part_pcs.shape  # [B, N_sum, 3]
        # shared-weight encoder
        valid_pcs = part_pcs.reshape(B * N_sum, -1)
        valid_feats = self.encoder(valid_pcs, batch_length)  # [B * N_sum, F]
        pc_feats = valid_feats.reshape(B, N_sum, -1)  # [B, N_sum, F]
        return pc_feats

    def _get_critical_feats_BNF_from_label(
            self, feat, n_critical_pcs_sum, critical_label, B, N_, F
    ):
        critical_feats = torch.zeros(B, N_, F, device=self.device, dtype=feat.dtype)
        for b in range(B):
            critical_feats[b, : n_critical_pcs_sum[b]] = feat[b, critical_label[b] == 1]
        return critical_feats

    def forward(self, data_dict):
        """
        Forward pass to predict matching
        :param data_dict:
                - part_pcs: [B, N_sum, 3]
                - part_feats: [B, N_sum, F] reused or None
                - critical_pcs_idx: [B, N'_sum]
                - n_critical_pcs: [B, P]
                - gt_pcs: [B, N_sum, 3]
                - part_valids: [B, P], 1 are valid parts, 0 are padded parts
        :return:
            out_dict:
                - ds_mat: [B, N', N'] a doubly stochastic matrix, differentiable matching result
                - perm_mat: [B, N', N'] sparse matching result
        """
        part_valids = data_dict["part_valids"]
        n_valid = torch.sum(part_valids, dim=1).to(torch.long)  # [B]
        n_pcs = data_dict["n_pcs"]  # [B, P]

        part_pcs = data_dict["part_pcs"]  # [B, N_sum, 3]
        part_input = part_pcs

        B, N_sum, _ = part_pcs.shape
        part_feats = data_dict.get("part_feats", None)  # [B, N_sum, F]
        batch_length = get_batch_length_from_part_points(n_pcs, n_valids=n_valid).to(
            self.device
        )

        if part_feats is None:
            part_feats = self._extract_part_feats(part_input, batch_length)
            part_pcs_flatten = part_pcs.reshape(-1, 3).contiguous()
            for name, layer in self.tf_layers:
                if name == "self":
                    part_feats = (
                        layer(
                            part_pcs_flatten,
                            part_feats.view(-1, self.pc_feat_dim),
                            batch_length,
                        )
                        .view(B, N_sum, -1)
                        .contiguous()
                    )
                else:
                    part_feats = layer(part_feats)
            data_dict.update({"part_feats": part_feats})
        out_dict = dict()
        feat = part_feats.transpose(1, 2)
        if self.pc_cls_method == 'binary':
            cls_logits = self.pc_classifier(feat)
            cls_logits = cls_logits.transpose(1, 2)
            cls_pred = torch.sigmoid(cls_logits.detach())
            cls_pred = (cls_pred > 0.5).to(torch.int64)
            cls_pred = cls_pred.reshape(B, N_sum)
        else:
            cls_logits = self.pc_classifier(feat)
            cls_logits = fun.log_softmax(cls_logits, dim=1)
            cls_logits = cls_logits.permute(0, 2, 1).contiguous()
            cls_pred = torch.argmax(cls_logits.reshape(-1, self.num_classes), dim=-1).detach().reshape(B, N_sum)
        out_dict.update(
            {
                "cls_logits": cls_logits,  # [B, N_sum, 1] or [B, N_sum, 2]
                "cls_pred": cls_pred,  # [B, N_sum]
                "batch_size": B,
            }
        )

        if (not self.training) and self.trainer.testing:
            # the label is only used in testing
            # if you want to use them in training, the number of critical points might be too large for V100.
            with torch.no_grad():
                critical_pcs_idx, n_critical_pcs = get_critical_pcs_from_label(
                    cls_pred, n_pcs
                )
                data_dict.update(
                    {
                        "critical_label": cls_pred,
                        "critical_pcs_idx": critical_pcs_idx,
                        "n_critical_pcs": n_critical_pcs,
                    }
                )
                # print("update critical_pcs based on prediction")  # a test time reminder

        n_critical_pcs = data_dict.get("n_critical_pcs", None)
        critical_label = data_dict.get("critical_label", None)
        if n_critical_pcs is None:
            # if the label doesn't exist, calc the ground truth
            with torch.no_grad():
                gt_pcs = data_dict["gt_pcs"]  # [B, N_sum, 3]
                critical_label_threshold = data_dict["critical_label_thresholds"]
                critical_label = self.compute_label(
                    gt_pcs, n_pcs, n_valid, critical_label_threshold
                )
                critical_pcs_idx, n_critical_pcs = get_critical_pcs_from_label(
                    critical_label, n_pcs
                )
                data_dict.update(
                    {
                        "critical_label": critical_label,
                        "n_critical_pcs": n_critical_pcs,
                        "critical_pcs_idx": critical_pcs_idx,
                    }
                )

        if self.training and self.w_mat_loss == 0:
            # shorten the training time
            return out_dict

        F = part_feats.shape[-1]

        n_critical_pcs_sum = torch.sum(n_critical_pcs, dim=-1)  # [B]
        N_ = torch.max(n_critical_pcs_sum)

        critical_feats = self._get_critical_feats_BNF_from_label(
            part_feats, n_critical_pcs_sum, critical_label, B, N_, F
        )

        affinity_feat = self.affinity_extractor(critical_feats.permute(0, 2, 1))
        affinity_feat = affinity_feat.permute(0, 2, 1)

        affinity_feat = torch.cat(
            [
                fun.normalize(
                    affinity_feat[:, :, : self.half_aff_feat_dim], p=2, dim=-1
                ),
                fun.normalize(
                    affinity_feat[:, :, self.half_aff_feat_dim:], p=2, dim=-1
                ),
            ],
            dim=-1,
        )

        s = self.affinity_layer(affinity_feat, affinity_feat)

        if (not self.training) and self.cfg.MODEL.TEST_S_MASK:
            mask = self.diagonal_square_mask(
                s.shape, n_critical_pcs, n_part=n_valid, pos_msk=1, neg_msk=0
            ).detach()
            neg_mask = self.diagonal_square_mask(
                s.shape, n_critical_pcs, n_part=n_valid, pos_msk=0, neg_msk=-1e6
            ).detach()
            s_ = s * mask + neg_mask
            out_dict.update(
                {
                    "s_mask": mask,
                    "s_neg_mask": neg_mask,
                }
            )
        else:
            s_ = s

        mat = self.sinkhorn(s_, n_critical_pcs_sum, n_critical_pcs_sum)
        out_dict.update(
            {
                "ds_mat": mat,  # [B, N_, N_]
            }
        )

        if not self.training:
            perm_mat = hungarian(mat, n_critical_pcs_sum, n_critical_pcs_sum)
            out_dict.update({"perm_mat": perm_mat})
        return out_dict

    def _loss_function(self, data_dict, out_dict={}, optimizer_idx=-1):
        gt_pcs = data_dict["gt_pcs"]
        part_pcs = data_dict["part_pcs"]
        B, N_sum, _ = gt_pcs.shape
        critical_pcs_idx = data_dict["critical_pcs_idx"]
        n_critical_pcs = data_dict["n_critical_pcs"]
        part_valids = data_dict["part_valids"]
        n_pcs = data_dict["n_pcs"]
        critical_label = data_dict["critical_label"]

        cls_logits = out_dict["cls_logits"]
        cls_pred = out_dict["cls_pred"]

        loss_dict = {
            "batch_size": B,
        }

        # first calc segmentation
        cls_gt = critical_label.reshape(-1)
        if self.pc_cls_method == 'binary':
            cls_logits = cls_logits.reshape(-1)
            cls_loss = fun.binary_cross_entropy_with_logits(cls_logits, cls_gt.to(torch.float32))
        else:
            cls_logits = cls_logits.reshape(-1, self.num_classes)
            cls_loss = fun.nll_loss(cls_logits, cls_gt)

        cls_pred = cls_pred.reshape(-1)

        cls_acc = torchmetrics.functional.accuracy(cls_pred, cls_gt, task="binary")
        cls_precision = torchmetrics.functional.precision(
            cls_pred, cls_gt, task="binary"
        )
        cls_recall = torchmetrics.functional.recall(cls_pred, cls_gt, task="binary")
        cls_f1_score = torchmetrics.functional.f1_score(cls_pred, cls_gt, task="binary")

        loss_dict.update(
            {
                "cls_loss": cls_loss,
                "cls_acc": cls_acc,
                "cls_precision": cls_precision,
                "cls_recall": cls_recall,
                "cls_f1": cls_f1_score,
            }
        )

        if self.training and self.w_mat_loss == 0:
            loss_dict.update({"loss": cls_loss})
            return loss_dict

        # calc matching ground truth
        n_valid = torch.sum(part_valids, dim=1).to(torch.int)
        n_critical_pcs_sum = torch.sum(n_critical_pcs, dim=-1)  # [B]
        N_ = torch.max(n_critical_pcs_sum)
        with torch.no_grad():
            gt_critical_pcs = self._get_critical_feats_BNF_from_label(
                gt_pcs, n_critical_pcs_sum, critical_label, B, N_, 3
            )  # [B, N_, 3]
            gt_critical_pcs_dist = square_distance(gt_critical_pcs, gt_critical_pcs)  # [B, N_, N_]
            mask = out_dict.get("s_mask", None)
            neg_mask = out_dict.get("s_neg_mask", None)
            if neg_mask is None:
                neg_mask = self.diagonal_square_mask(
                    shape=(B, N_, N_),
                    n_pcs=n_critical_pcs,
                    n_part=n_valid,
                    pos_msk=0,
                    neg_msk=-1e6,
                )
            if mask is None:
                mask = self.diagonal_square_mask(
                    shape=(B, N_, N_),
                    n_pcs=n_critical_pcs,
                    n_part=n_valid,
                    pos_msk=1,
                    neg_msk=0,
                )
            gt_critical_pcs_dist -= neg_mask
            gt_pcs_dis_min_idx = torch.argmin(gt_critical_pcs_dist, dim=-1).reshape(
                B, N_, -1
            )  # [B, N_]

            gt_perm = torch.zeros(B, N_, N_, device=self.device).scatter_(
                2, gt_pcs_dis_min_idx, 1
            )
            gt_perm *= mask

        mat = out_dict["ds_mat"]

        # calc matching loss
        mat_loss = permutation_loss(
            mat, gt_perm, n_critical_pcs_sum, n_critical_pcs_sum
        )

        loss_dict.update(
            {
                "mat_loss": mat_loss,
                "N_": N_,
            }
        )
        if self.w_rig_loss > 0:
            rig_loss = rigid_loss(
                n_critical_pcs, mat, gt_pcs, critical_pcs_idx, part_pcs, n_valid, n_pcs
            )
            loss_dict.update({"rig_loss": rig_loss})
        else:
            rig_loss = 0

        if self.training:
            loss = (
                    self.w_cls_loss * cls_loss
                    + self.w_mat_loss * mat_loss
                    + self.w_rig_loss * rig_loss
            )
        else:
            loss = cls_loss + mat_loss

        loss_dict.update(
            {
                "loss": loss,
            }
        )

        # calc matching metric to monitor training process
        perm_mat = out_dict.get("perm_mat", None)
        if perm_mat is not None:
            tp, fp, fn = 0, 0, 0
            for b in range(B):
                pred = perm_mat[b, : n_critical_pcs_sum[b], : n_critical_pcs_sum[b]]
                gt_pred = gt_perm[b, : n_critical_pcs_sum[b], : n_critical_pcs_sum[b]]
                tp += torch.sum(pred * gt_pred).float()
                fp += torch.sum(pred * (1 - gt_pred)).float()
                fn += torch.sum((1 - pred) * gt_pred).float()
            const = torch.tensor(1e-7, device=self.device)
            precision = tp / (tp + fp + const)
            recall = tp / (tp + fn + const)
            f1 = 2 * precision * recall / (precision + recall + const)
            loss_dict.update(
                {
                    "mat_f1": f1,
                    "mat_precision": precision,
                    "mat_recall": recall,
                }
            )
        
        # self._save_perm_mat(data_dict, perm_mat, critical_pcs_idx, n_critical_pcs, gt_pcs, n_pcs)

        return loss_dict
    
    # def _save_perm_mat(self, data_dict, perm_mat, 
    #                    critical_pcs_idx, n_critical_pcs, 
    #                    gt_pcs, n_pcs,
    #                    save_dir="perm_mat/v2"):
    #     data_id = data_dict["data_id"]
    #     os.makedirs(save_dir, exist_ok=True)
    #     for i in range(len(data_id)):
    #         perm_mat_i = perm_mat[i].detach().cpu().numpy()
    #         critical_pcs_idx_i = critical_pcs_idx[i].detach().cpu().numpy()
    #         n_critical_pcs_i = n_critical_pcs[i].detach().cpu().numpy()
    #         gt_pcs_i = gt_pcs[i].detach().cpu().numpy()
    #         n_pcs_i = n_pcs[i].detach().cpu().numpy()
            
    #         save_dict = {
    #             "perm_mat": perm_mat_i.astype(np.float32), 
    #             "critical_pcs_idx": critical_pcs_idx_i, 
    #             "n_critical_pcs": n_critical_pcs_i,
    #             "gt_pcs": gt_pcs_i.astype(np.float32),
    #             "n_pcs": n_pcs_i
    #         }

    #         np.savez(os.path.join(save_dir, str(data_id[i].item())), **save_dict)
            
        

    def training_epoch_end(self, outputs):
        if self.w_mat_loss == 0 and self.current_epoch >= self.cfg.MODEL.LOSS.mat_epoch:
            self.w_mat_loss = 1.0
            print(
                f"current_epoch={self.current_epoch}, self.w_mat_los = 1.0"
            )
        if self.w_rig_loss == 0 and self.current_epoch >= self.cfg.MODEL.LOSS.rig_epoch:
            self.w_rig_loss = 1.0
            print(
                f"current_epoch={self.current_epoch}, self.w_rig_loss = 1.0"
            )

    @torch.no_grad()
    def compute_label(self, part_pcs, nps, n_valid, label_thresholds):
        """
        Compute ground truth label of fracture points.
        :param part_pcs: all points from all pieces, [B, N_sum, 3]
        :param nps: number of points for each piece, [B, N]
        :param n_valid: number of valid parts in each object, [B]
        :param label_thresholds: threshold for ground truth label, [B, N_sum]
        :return: labels: 1 if point is a fracture point and 0 otherwise [B, N_sum]
        """
        B, N_, _ = part_pcs.shape
        dists = torch.sqrt(square_distance(part_pcs, part_pcs))
        neg_mask = self.diagonal_square_mask(
            shape=(B, N_, N_), n_pcs=nps, n_part=n_valid, pos_msk=0, neg_msk=1e6
        )
        dists = dists + neg_mask
        dists_min, _ = torch.min(dists, dim=-1)
        dists_min = dists_min.reshape(B, N_)
        labels = (dists_min < label_thresholds).to(torch.int64)
        return labels

    @torch.no_grad()
    def diagonal_square_mask(
            self, shape, n_pcs, n_part=None, pos_msk=0.0, neg_msk=1000.0
    ):
        """
        generate a mask which diagonal matrices are neg_msk and others pos_mask
        :param shape: list like [B, N_, N_], the shape of wanted mask
        :param n_pcs: [B, P] points of each part
        :param n_part: [B] number of parts of each object
        :param pos_msk: positive mask
        :param neg_msk: negative mask
        :return: msk: a matrix mask out diagonal squares with neg_msk.
        """
        # shape [B, N_, N_]
        B = n_pcs.shape[0]
        n_pcs_cumsum = torch.cumsum(n_pcs, dim=-1)  # [B, P]
        if n_part is None:
            P = n_pcs_cumsum.shape[-1]
            n_part = torch.tensor([P for _ in range(B)], dtype=torch.long)
        msk = torch.ones(shape).to(self.device) * neg_msk
        for b in range(B):
            n_p = n_part[b]
            msk[b, : n_pcs_cumsum[b, n_p - 1], : n_pcs_cumsum[b, n_p - 1]] = pos_msk
            for p in range(n_part[b]):
                st = 0 if p == 0 else n_pcs_cumsum[b, p - 1]
                ed = n_pcs_cumsum[b, p]
                msk[b, st:ed, st:ed] = neg_msk
        return msk
