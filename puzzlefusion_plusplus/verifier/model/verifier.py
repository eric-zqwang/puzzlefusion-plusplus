import lightning.pytorch as pl
import torch
from puzzlefusion_plusplus.verifier.model.modules.verifier_transformer import VerifierTransformer
import torch.nn.functional as fun
import torchmetrics


class Verifier(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.verifier = VerifierTransformer(cfg)

    def forward(self, edge_features, edge_indices, edge_valids):
        logits = self.verifier(edge_features, edge_indices, edge_valids)
        output_dict = {"logits": logits}
        return output_dict

    def _loss(self, data_dict, output_dict):
        logits = output_dict["logits"].reshape(-1)
        cls_gt = data_dict["cls_gt"].reshape(-1).float()
        edge_valids = data_dict["edge_valids"].reshape(-1).to(torch.bool)
        logits = logits[edge_valids]
        cls_gt = cls_gt[edge_valids]

        cls_loss = fun.binary_cross_entropy_with_logits(logits, cls_gt, weight=torch.where(cls_gt==0, 0.2, 1.0))

        cls_pred = torch.sigmoid(logits.detach())
        cls_pred = (cls_pred > 0.5).to(torch.int64)

        cls_acc = torchmetrics.functional.accuracy(cls_pred, cls_gt, task="binary")
        cls_precision = torchmetrics.functional.precision(
            cls_pred, cls_gt, task="binary"
        )
        cls_recall = torchmetrics.functional.recall(cls_pred, cls_gt, task="binary")
        cls_f1_score = torchmetrics.functional.f1_score(cls_pred, cls_gt, task="binary")

        loss_dict = {
            "cls_acc": cls_acc,
            "cls_precision": cls_precision,
            "cls_recall": cls_recall,
            "cls_f1_score": cls_f1_score,
            "cls_loss": cls_loss,
        }

        return loss_dict

    def training_step(self, data_dict, idx):
        edge_features = data_dict["edge_features"]
        edge_indices = data_dict["edge_indices"]
        edge_valids = data_dict["edge_valids"]

        output_dict = self(edge_features, edge_indices, edge_valids)

        loss_dict = self._loss(data_dict, output_dict)
        total_loss = loss_dict["cls_loss"]

        cls_precision = loss_dict["cls_precision"]
        cls_recall = loss_dict["cls_recall"]
        cls_f1_score = loss_dict["cls_f1_score"]
        cls_acc = loss_dict["cls_acc"]
        self.log(f"training/loss", total_loss, on_step=True, on_epoch=False)
        self.log(f"training/cls_precision", cls_precision, on_step=False, on_epoch=True)
        self.log(f"training/cls_recall", cls_recall, on_step=False, on_epoch=True)
        self.log(f"training/cls_f1_score", cls_f1_score, on_step=False, on_epoch=True)
        self.log(f"training/cls_acc", cls_acc, on_step=False, on_epoch=True)

        return total_loss


    def validation_step(self, data_dict, idx):
        edge_features = data_dict["edge_features"]
        edge_indices = data_dict["edge_indices"]
        edge_valids = data_dict["edge_valids"]

        output_dict = self(edge_features, edge_indices, edge_valids)

        loss_dict = self._loss(data_dict, output_dict)
        total_loss = loss_dict["cls_loss"]

        cls_precision = loss_dict["cls_precision"]
        cls_recall = loss_dict["cls_recall"]
        cls_f1_score = loss_dict["cls_f1_score"]
        cls_acc = loss_dict["cls_acc"]

        self.log(f"val/loss", total_loss, on_step=False, on_epoch=True)
        self.log(f"val/cls_precision", cls_precision, on_step=False, on_epoch=True)
        self.log(f"val/cls_recall", cls_recall, on_step=False, on_epoch=True)
        self.log(f"val/cls_f1_score", cls_f1_score, on_step=False, on_epoch=True)
        self.log(f"val/cls_acc", cls_acc, on_step=False, on_epoch=True)


    def test_step(self, data_dict, idx):
        self.validation_step(data_dict, idx)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=2e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )
        return optimizer

