python train_denoiser.py \
    experiment_name=everyday_epoch2000_bs64 \
    data.batch_size=64 \
    data.val_batch_size=64 \
    model.encoder_weights_path=output/autoencoder/everyday_2000epoch/training/last.ckpt \
    +trainer.devices=4 \
    +trainer.strategy=ddp