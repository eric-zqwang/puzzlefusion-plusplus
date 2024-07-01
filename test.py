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
    model = AutoAgglomerative(cfg)

    denoiser_weights = torch.load(cfg.denoiser.ckpt_path)['state_dict']

    model.denoiser.load_state_dict(
        {k.replace('denoiser.', ''): v for k, v in denoiser_weights.items() 
         if k.startswith('denoiser.')}
    )

    model.encoder.load_state_dict(
        {k.replace('encoder.', ''): v for k, v in denoiser_weights.items() 
         if k.startswith('encoder.')}
    )

    # load verifier weights    
    verifier_weights = torch.load(cfg.verifier.ckpt_path)['state_dict']
    model.verifier.load_state_dict({k.replace('verifier.', ''): v for k, v in verifier_weights.items()})
    # initialize trainer
    trainer = pl.Trainer(accelerator=cfg.accelerator, max_epochs=1, logger=False)
    
    # start inference
    trainer.test(model=model, dataloaders=test_loader)


if __name__ == '__main__':
    main()
