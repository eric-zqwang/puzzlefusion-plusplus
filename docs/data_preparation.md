## Data preparation
We follow the 
[Breaking Bad Dataset](https://breaking-bad-dataset.github.io/) for data pre-processing.
For more information about data processing, please refer to the dataset website.

After processing the data, ensure that you have a folder named `data` with the following structure:
```
data
├── breaking_bad
│   ├── everyday
│   │   ├── BeerBottle
│   │   │   ├── ...
│   │   ├── ...
│   ├── everyday.train.txt
│   ├── everyday.val.txt
│   └── ...
└── ...
```
Only the `everyday` subset is necessary.

### Generate point cloud data
In the orginal benchmark code of Breaking Bad dataset, it needs sample point cloud from mesh in each batch which is time-consuming. We pre-processing the mesh data and generate its point cloud data and its attribute.
```
cd puzzlefusion-plusplus/
python generate_pc_data +data save_pc_data_path=data/pc_data/everyday/
```

### Verifier training data
You can download the verifier data from [here](https://1sfu-my.sharepoint.com/:f:/g/personal/zwa170_sfu_ca/EtSHHinoDndPs8kJfRn_n0QBue1ypoXGkNEOio9pU6bFcQ?e=pkcuox).

### Matching data
You can download the matching data from [here](https://1sfu-my.sharepoint.com/:f:/g/personal/zwa170_sfu_ca/EtSHHinoDndPs8kJfRn_n0QBue1ypoXGkNEOio9pU6bFcQ?e=pkcuox).

The verifier data and matching data need to generate the data from [Jigsaw](https://github.com/Jiaxin-Lu/Jigsaw). Since this process is quite complex, we will upload the processed data for now. More details on how to obtain this processed data will be provided later.

## Checkpoints
We provide the checkpoints at this [link](https://1sfu-my.sharepoint.com/:f:/g/personal/zwa170_sfu_ca/EoYp5Z5WiqtNuq_GOb5Yj1ABSI5lQSXG64StzXb6eTbXNg?e=N3uJ7L). Please download and place them as ./work_dirs/ then unzip.

## Structure
Finally, the overall data structure should looks like:
```
puzzlefusion-plusplus/
├── data
│   ├── pc_data
│   ├── verifier_data
│   ├── matching_data
└── ...
├── output
│   ├── autoencoder
│   ├── denoiser
│   ├── ...
└── ...
```
