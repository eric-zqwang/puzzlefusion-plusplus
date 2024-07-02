## Test
We provide the our checkpoints in [data preparation](../docs/data_preparation.md).
In addition, you need make sure you download the matching data from [data preparation](../docs/data_preparation.md).
You need modify the checkpoint path for both pre-trained denoiser and verifier in the script. 
```
sh ./scripts/inference.sh
```
We only support batch size equal to one for testing.
The inference results of pose parameter are stored in ./output/denoiser/{experiemnt_name}/inference/{inference_dir}. You can use these saved results to do visualization later.

## Evaluation


## Visualization
