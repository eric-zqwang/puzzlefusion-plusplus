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
By running with the checkpoints we provided in the [data preparation](../docs/data_preparation.md) guide, the expected results are:
```
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      eval/part_acc         0.7018406391143799
       eval/rmse_r           38.46787643432617
       eval/rmse_t          0.07968249917030334
      eval/shape_cd         0.0065745091997087
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

## Visualization
We will upload visualization code as soon as possible.