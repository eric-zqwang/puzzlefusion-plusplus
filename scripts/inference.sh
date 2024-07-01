python test.py \
    experiment_name=everyday_epoch2000_gpu4_bs64 \
    denoiser.data.val_batch_size=1 \
    denoiser.data.data_val_dir=./data/pc_data/everyday/val/ \
    denoiser.ckpt_path=output/denoiser/everyday_epoch2000_gpu4_bs64/training/last.ckpt \
    verifier.ckpt_path=output/verifier/everyday_epoch100_bs64/last.ckpt \
    denoiser.data.matching_data_path=./data/matching_data/ \
    inference_dir=results \
    verifier.max_iters=6 \
