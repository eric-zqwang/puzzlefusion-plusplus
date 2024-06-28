import torch
import hydra
import lightning.pytorch as pl
import copy
import math

class FractureAE(pl.LightningModule):
    def __init__(self, cfg):
        super(FractureAE, self).__init__()
        self.ae = hydra.utils.instantiate(cfg.ae.ae_name, cfg)
        self.cfg = cfg

    def forward(self, data_dict):
        original_data_dict = copy.deepcopy(data_dict)

        part_pcs = data_dict['part_pcs']
        num_parts = data_dict['num_parts']

        B, N, _, _ = part_pcs.shape

        range_tensor = torch.arange(N).unsqueeze(0).expand(B, -1).to(part_pcs.device)
        mask = range_tensor < num_parts.unsqueeze(1)
        part_pcs = part_pcs[mask]

        data_dict['part_pcs'] = part_pcs

        data_dict["iters"] = self.trainer.global_step
        
        output_dict = self.ae(data_dict)

        return output_dict, original_data_dict




    def _loss(self, data_dict, output_dict):

        loss_dict = self.ae.loss(data_dict, output_dict)

        return loss_dict
 

    def training_step(self, data_dict, idx):
        output_dict, _ = self(data_dict)

        # if output dict have perplexity, then self.log it
        if "perplexity" in output_dict.keys():
            self.log("train_perplexity", output_dict["perplexity"], on_step=True, on_epoch=False)

        loss_dict = self._loss(data_dict, output_dict)

        total_loss = 0
        for loss_name, loss_value in loss_dict.items():
            total_loss += loss_value
            self.log(f"train_loss/{loss_name}", loss_value, on_step=True, on_epoch=False)
        self.log(f"train_loss/total_loss", total_loss, on_step=True, on_epoch=False)
    
        return total_loss
    

    def validation_step(self, data_dict, idx):
        output_dict, _ = self(data_dict)
        loss_dict = self._loss(data_dict, output_dict)
        
        total_loss = 0

        for loss_name, loss_value in loss_dict.items():
            total_loss += loss_value
            self.log(f"val_loss/{loss_name}", loss_value, on_step=False, on_epoch=True)
        self.log(f"val_loss/total_loss", total_loss, on_step=False, on_epoch=True)
        
    

    def test_step(self, data_dict, idx):
        output_dict, original_data_dict = self(data_dict)

        
    def on_test_epoch_end(self):
        pass


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=5e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )
        lr_scheduler = hydra.utils.instantiate(self.cfg.model.lr_scheduler, optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
