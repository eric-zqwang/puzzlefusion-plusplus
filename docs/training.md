## Training

The training consists of three modules as detailed in the paper. We train the vqvae and denoiser on 4 Nvidia RTX A6000 GPUs. The verifier is trained on a single RTX 4090 GPU.

**Stage 1**: VQVAE:
```
sh ./scripts/train_vqvae.sh
```

**Stage 2**: SE3 denoiser:
```
sh ./sripts/train_denoiser.sh
```
You need modify the checkpoint path for the pre-trained VQVAE in the script.

**Stage 3**: Pairwise alignment verifier:
```
sh ./sripts/train_verifier.sh
```
