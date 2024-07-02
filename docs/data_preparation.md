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
We also provide the pre-processed data in [here](https://drive.google.com/file/d/1nCG18WEDuy2LoYt2pyt6UUgAZrsp4xWf/view?usp=drive_link).

### Verifier training data
You can download the verifier data from [here](https://drive.google.com/file/d/19qjY8pbftUK70tJWqd3xfBNrzMUygq4G/view?usp=drive_link).

### Matching data
You can download the matching data from [here](https://drive.google.com/file/d/1wC9BU_Z2PD8G7UkPczZDFuDZhQnuKKvp/view?usp=drive_link).

The verifier data and matching data need to generate the data from [Jigsaw](https://github.com/Jiaxin-Lu/Jigsaw). Since this process is quite complex, we will upload the processed data for now. More details on how to obtain this processed data will be provided later.

## Checkpoints
We provide the checkpoints at this [link](https://drive.google.com/file/d/1oj2t7nwRJMizGveBANR21ssF4o3TNJ8Q/view?usp=sharing). Please download and place them as ./work_dirs/ then unzip.

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
