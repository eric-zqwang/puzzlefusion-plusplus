python train_vqvae.py \
    experiment_name=everyday_2000epoch \
    data.batch_size=45 \
    data.val_batch_size=45 \
    +trainer.devices=4 \
    +trainer.strategy=ddp