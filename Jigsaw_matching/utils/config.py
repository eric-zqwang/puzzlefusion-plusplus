import importlib

from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.MODEL_NAME = ""  # this name would be the result file name
__C.MODULE = ""  # sample: dgl.network  b_global.network

__C.BATCH_SIZE = 32
__C.NUM_WORKERS = 8

__C.LOG_FILE_NAME = ""  # the suffix of log file

__C.MODEL_SAVE_PATH = ""  # auto generated

#
# Dataset
#
__C.DATASET = ""

# Other dataset specific configs should be imported from dataset_config.py

# wandb project name
__C.PROJECT = ""

#
# Training options
#

__C.TRAIN = edict()

# Total epochs
__C.TRAIN.NUM_EPOCHS = 200

# Optimizer type
__C.TRAIN.OPTIMIZER = "SGD"

# Start learning rate
__C.TRAIN.LR = 0.001

# LR Scheduler
__C.TRAIN.LR_SCHEDULER = "cosine"

# Learning rate decay
__C.TRAIN.LR_DECAY = 100.0

# Learning rate decay step (in epochs)
__C.TRAIN.LR_STEP = [10, 20]

# warmup_ratio for Adam Cosine
__C.TRAIN.WARMUP_RATIO = 0.0

# clip_grad
__C.TRAIN.CLIP_GRAD = None

# beta1, beta2 for Adam Optimizer
__C.TRAIN.beta1 = 0
__C.TRAIN.beta2 = 0.9

# weight decay for Adam or SGD
__C.TRAIN.WEIGHT_DECAY = 0.0

# SGD momentum
__C.TRAIN.MOMENTUM = 0.9

# Check val every n epoch
__C.TRAIN.VAL_EVERY = 5

# Visualization during training
__C.TRAIN.VIS = True
__C.TRAIN.VAL_SAMPLE_VIS = 5

# Loss function.
__C.TRAIN.LOSS = ""

#
# Callback
#
__C.CALLBACK = edict()
__C.CALLBACK.MATCHING_TASK = ["trans"]
__C.CALLBACK.CHECKPOINT_MONITOR = "val/loss"
__C.CALLBACK.CHECKPOINT_MODE = "min"

#
# Loss config
#
__C.LOSS = edict()

#
# Evaluation options
#

__C.EVAL = edict()

#
# MISC
#

# Parallel GPU indices ([0] for single GPU)
__C.GPUS = [0]
# Parallel strategy for multiple gpus
__C.PARALLEL_STRATEGY = "ddp"

# Float Precision, 32 for False, 16 for True
__C.FP16 = False

# CUDNN benchmark
__C.CUDNN = False

__C.WEIGHT_FILE = ""

# Output path (for checkpoints, running logs)
__C.OUTPUT_PATH = ""

# The step of iteration to print running statistics.
# The real step value will be the least common multiple of this value and batch_size
__C.STATISTIC_STEP = 100

# random seed used for data loading
__C.RANDOM_SEED = 42

# directory for collecting statistics of results
__C.STATS = ""


def _merge_a_into_b(a, b):
    """Merge config dictionary A into config dictionary B, clobbering the
    options in B whenever they are also specified in A.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError("{} is not a valid config key".format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            if type(b[k]) is float and type(v) is int:
                v = float(v)
            else:
                if k not in ["CLASS"]:
                    raise ValueError(
                        "Type mismatch ({} vs. {}) for config key: {}".format(
                            type(b[k]), type(v), k
                        )
                    )

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print("Error under config key: {}".format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml

    with open(filename, "r") as f:
        yaml_cfg = edict(yaml.full_load(f))

    # DONE: for model specific or data specific configs, please define them in the model package
    #  and then load them here
    if "MODULE" in yaml_cfg and "MODEL" not in __C:
        model_cfg_module = ".".join(
            ["model"] + [yaml_cfg.MODULE.split(".")[0]] + ["model_config"]
        )
        # model_cfg_module = 'model.model_config'
        mod = importlib.import_module(model_cfg_module)
        __C["MODEL"] = mod.get_model_cfg()

    if "DATASET" in yaml_cfg and yaml_cfg.DATASET is not None:
        dataset = importlib.import_module("dataset")
        __C["DATA"] = dataset.dataset_cfg[yaml_cfg.DATASET.split(".")[0].upper()]

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval

    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split(".")
        d = __C
        for sub_key in key_list[:-1]:
            assert sub_key in d.keys()
            d = d[sub_key]
        sub_key = key_list[-1]
        assert sub_key in d.keys()
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(
            d[sub_key]
        ), "type {} does not match original type {}".format(
            type(value), type(d[sub_key])
        )
        d[sub_key] = value
