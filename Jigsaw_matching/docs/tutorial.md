# Tutorial

In this tutorial, we invite you to an exploration of this codebase.
Our hope is to introduce the intricacies of Jigsaw models and offer
some insights of the training process.

## Code Structure

Let's dive into the organization structure that forms the backbone of our work.
```
Jigsaw
├── dataset
│   ├── __init__.py
│   ├── dataset_and_loaders.py
│   └── dataset_config.py
├── experiments
│   └── your_configs.yaml
├── model
│   ├── jigsaw
│   │   ├── __init__.py
│   │   ├── model_config.py
│   │   ├── joint_seg_align_model.py
│   │   └── ...
│   ├── modules
│   │   ├── encoder
│   │   │   ├── pointnet2_pointwise
│   │   │   │   ├── ...
│   │   │   ├── dgcnn.py
│   │   │   └── ...
│   │   ├── matching_base_model.py
│   │   └── __init__.py
│   └── __init__.py
├── utils
│   └── supporting_files.py
├── train_matching.py
└── eval_matching.py
```

## Dataset

Within the `dataset` directory, you'll find a collection of customized datasets and
data loaders catering to the fracture assembly task.

A central component of the dataset is our `AllPieceMatchingDataset`.
The dataset is tailored for our matching-based technique.
A detailed documentation of the returned `data_dict` can be found in accompanying comments.
Different from the Breaking Bad Dataset benchmark code,
we support point sampling based on piece surface area. 
There are subtle adjustments introduced to the data tensor structure.
Our implementation can guarantee a minimum sampling threshold for each piece, via the `MIN_PART_POINT`
configuration, through a greedy algorithm.

For a deeper understanding of dataset configurations, please refer
to the `dataset_config.py` file.

For now, we only support our dataset with the `area` sampling strategy.

## Model

### Modules

We implemented a `MatchingBaseModel` for better supporting matching-based
assembly models. Building upon the `BaseModel` established by the Breaking Bad Dataset
benchmark code, we introduce global alignment post-processing schemes to assist testing.

Our support extend to integrating PointNet++ and DGCNN to accommodate dynamic point
numbers influenced by our sampling strategy. You can access them by specifying config
by `{method}.dynamic`.

### Jigsaw

The fundamental model architectures are included in this folder.
The overall model is presented in `joint_seg_align_model.py`. 
Primal-dual descriptors and affinity metric are in `affinity_layer.py`.
The `attention_layer.py` includes the self-attention and cross-attention layers.
Model-specific configs can be accessed in `model_config.py`.

The matching loss and rigidity loss we used are implemented in `utils/loss.py`
`Sinkhorn` and `Hungarian` can be found in `utils/linear_solvers.py`. We thank
ThinkMatch and Pygmtools for the source code.

As a side note, we implement two segmentation strategy based on binary and multi-class 
classification for now. Our recent empirical findings indicate that
the latter version is slightly (~1.0 improvement in R metrics) better. 
You can find both versions of checkpoints in our release.

## Config System

We want to introduce the config system so that you may design your own 
config files! Our config system is build upon `yaml` and `EasyDict` to provide
flexibility while maintaining a clear focus on your unique needs.

The corner of config system is `utils/config.py`, where contains configurations
governing training and testing. 
As you craft your config files, use the following template as a guide:
```yaml
MODEL_NAME: your_model_name  # this will be the folder name of your training / testing
MODULE: module_name.branch  # should be a model registered in model/__init__.build_model()

PROJECT: wandb_project_name
DATASET: dataset_name.branch # should be a dataset registered in dataset/__init__.build_dataloader()

GPUS: [0, 1]  # list, gpus you want to use
BATCH_SIZE: batch_size
NUM_WORKERS: num_workers

TRAIN:
  NUM_EPOCH: epochs_to_train
  LR: learning_rate
  
CALLBACK:
  # All callback-related configurations reside here

DATA:
  # See below

MODEL:
  # See below

WEIGHT_FILE: ./path/to/your/checkpoint  # if you want to specifically resume from some checkpoint

OUTPUT_PATH: ./results/{your_model_name}  # automatically set
MODEL_SAVE_PATH: ./results/{your_model_name}/model_save  # automatically set
```

The dataset-specific configs shall be located in `dataset/dataset_config.py`, 
which can be accessed through `cfg.DATA` in your config file or model. 
Please make sure your `cfg.DATASET` align with `__C.{upper_case_dataset_name}`.

Method-specific configs are housed in `model/your_method/model_config.py`. 
You may access them in your config file or model by calling `cfg.MODEL`. 
Please also make sure your `cfg.MODULE` align with `__C.{upper_case_module_name}`. 
Here is a snippet of what your `model_config.py` should look like:

```python
from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

__C.UPPER_CASE_MODULE_NAME = edict()
__C.UPPER_CASE_MODULE_NAME.ENCODER = 'pointnet2_pt.msg.dynamic'

__C.UPPER_CASE_MODULE_NAME.LOSS = edict()
__C.UPPER_CASE_MODULE_NAME.LOSS.w1 = 1.0

def get_model_cfg():
    return model_cfg.UPPER_CASE_MODULE_NAME
```

Please remember that each configuration should be assigned with an initialization.
You can change any config through a config file `experiments/your_config.yaml`. 
If configurations aren't explicitly defined, the system defaults to their initialized values. 
Keep in mind that the value types in your config must match their corresponding initializations.

A sample config file for training Jigsaw can be found in the `experiments` folder.
We open source the config file along with the model checkpoints. You can find a training log on
this [wandb page](https://api.wandb.ai/links/assembly/u5s0xykd).


## PS

* Our minimal implementation involves real-time computation of ground truth segmentation
during training iterations, introducing a noticeable (almost double) time overhead.
A practical solution involves pre-saving point clouds and labels. 
Our results indicate that this will not impact outcomes while speeding up training.
You will need to change the dataset (but no need for model) to achieve this.

* As we move forward, we are committed to enriching this codebase to further support
research for assembly. Our plan involves expanding the range of supported datasets,
encoders, models, visualization, and more supporting scripts! Please let us know
which part should we prioritize.

