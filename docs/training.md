## Training

The training consists of three modules as detailed in the paper. We train the models on 4 Nvidia RTX A6000 GPUs. 

**Stage 1**: VQVAE:
```
sh ./scripts/train_vqvae.sh
```

**Stage 2**: SE3 denoiser:
```
sh ./sripts/train_denoiser.sh
```

**Stage 3**: Pairwise alignment verifier:
```
sh ./sripts/train_verifier.sh
```

We also have provided checkpoint for easier testing [here]().