## Test
We provide the our checkpoints in [data preparation](../docs/data_preparation.md).
You need make sure you download all the data from [data preparation](../docs/data_preparation.md).
We only support batch size equal to one for testing. You need modify the checkpoint path for both pre-trained denoiser and verifier in the script. 
```
sh ./scripts/inference.sh
```

The denoising parameter is stored in ./output/denoiser/{experiemnt_name}/inference/{inference_dir}. You can use this saved results to do visualization later.

[Jigsaw](https://github.com/Jiaxin-Lu/Jigsaw) uses sampling by area to generate point cloud data. The point cloud is created using their method, and the matching points are obtained from their network.

