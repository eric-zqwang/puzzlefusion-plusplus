import os
import hydra
import lightning.pytorch as pl
from puzzlefusion_plusplus.denoiser.dataset.dataset import build_test_dataloader
import torch
from puzzlefusion_plusplus.auto_aggl import AutoAgglomerative


@hydra.main(version_base=None, config_path="config", config_name="auto_aggl")
def main(cfg):
    # fix the seed
    pl.seed_everything(cfg.test_seed, workers=True)

    # create directories for inference outputs
    inference_dir = os.path.join(cfg.experiment_output_path, "inference", cfg.inference_dir)
    os.makedirs(inference_dir, exist_ok=True)

    # initialize data
    test_loader = build_test_dataloader(cfg.denoiser)

    # load denoiser weights
    model = AutoAgglomerative.load_from_checkpoint(cfg.denoiser.ckpt_path, cfg=cfg, strict=False)

    # load verifier weights    
    verifier_weights = torch.load(cfg.verifier.ckpt_path)['state_dict']
    model.verifier.load_state_dict({k.replace('verifier.', ''): v for k, v in verifier_weights.items()})

    # initialize trainer
    trainer = pl.Trainer(accelerator=cfg.trainer.accelerator, max_epochs=1, logger=False)
    
    # check the checkpoint
    assert cfg.ckpt_path is not None, "Error: Checkpoint path is not provided."
    assert os.path.exists(cfg.ckpt_path), f"Error: Checkpoint path {cfg.ckpt_path} does not exist."

    # start inference
    trainer.test(model=model, dataloaders=test_loader)


if __name__ == '__main__':
    main()
