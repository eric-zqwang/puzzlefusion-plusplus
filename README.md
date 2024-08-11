<div align="center">
<h1 align="center"> PuzzleFusion++: Auto-agglomerative 3D Fracture <br/> Assembly by Denoise and Verify
</h1>

### [Zhengqing Wang*<sup>1</sup>](https://eric-zqwang.github.io/) , [Jiacheng Chen*<sup>1</sup>](https://jcchen.me) , [Yasutaka Furukawa<sup>1,2</sup>](https://www2.cs.sfu.ca/~furukawa/)

### <sup>1</sup> Simon Fraser University <sup>2</sup> Wayve

### [arXiv](https://arxiv.org/abs/2406.00259), [Project page](https://puzzlefusion-plusplus.github.io/)
</div>


https://github.com/user-attachments/assets/37ec2a37-b88e-4de4-92af-e8a41756d7ff


This repository provides the official implementation of the paper [PuzzleFusion++: Auto-agglomerative 3D Fracture Assembly by Denoise and Verify](https://arxiv.org/abs/2406.00259).


## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Getting started](#getting-started)
- [Citation](#citation)
- [License](#license)

## Introduction

This paper proposes a novel “auto-agglomerative” 3D fracture assembly method, PuzzleFusion++, resembling how humans solve challenging spatial puzzles.

<div align="center">
<img src="docs/fig/arch.png" width=80% height=80%>
</div>

Starting from individual fragments, the approach 1) aligns and merges fragments into larger groups akin to agglomerative clustering and 2) repeats the process iteratively in completing the assembly akin to auto-regressive methods. Concretely, a diffusion model denoises the 6-DoF alignment parameters of the fragments simultaneously (the **Denoiser** in the figure above), and a transformer model verifies and merges pairwise alignments into larger ones (the **Verifier** in the figure above), whose process repeats iteratively.

Extensive experiments on the Breaking Bad dataset show that PuzzleFusion++ outperforms all other state-of-the-art techniques by significant margins across all metrics. In particular by over 10% in part accuracy and 50% in Chamfer distance.


## Installation

Please refer to the [installation guide](docs/installation.md) to set up the environment.


## Data preparation

Please refer to the [data preparation guide](docs/data_preparation.md) to download and prepare for the BreakingBad dataset, as well as downloading our pre-trained model checkpoints.


## Getting started

Please follow the [test guide](docs/test.md) for model inference, evaluation, and visualization.

Please follow the [training guide](docs/training.md) for details about the training pipeline.


## Citation

If you find PuzzleFusion++ useful in your research or applications, please consider citing:

```
@article{wang2024puzzlefusionpp,
  author    = {Wang, Zhengqing and Chen, Jiacheng and Furukawa, Yasutaka},
  title     = {PuzzleFusion++: Auto-agglomerative 3D Fracture Assembly by Denoise and Verify},
  journal   = {arXiv preprint arXiv:2406.00259},
  year      = {2024},
}
```

Our method is deeply inspired by [PuzzleFusion](https://github.com/sepidsh/PuzzleFussion) and [Jigsaw](https://github.com/Jiaxin-Lu/Jigsaw), and benefited from their open-source code. Please consider reading these papers if interested in relevant topics.


## License

This project is licensed under GPL, see the [license file](LICENSE) for details.
