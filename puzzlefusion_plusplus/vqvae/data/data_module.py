import lightning.pytorch as pl
from puzzlefusion_plusplus.vqvae.dataset.pc_dataset import build_geometry_dataloader


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_data, self.val_data = build_geometry_dataloader(cfg)

    def train_dataloader(self):
        return self.train_data

    def val_dataloader(self):
        return self.val_data

