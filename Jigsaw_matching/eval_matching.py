import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from dataset import build_dataloader
from model import build_model
from model.jigsaw.joint_seg_align_model import JointSegmentationAlignmentModel

NOW_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def test_model(cfg):
    if len(cfg.GPUS) > 1:
        raise ValueError("only one GPU testing is allowed")
    if len(cfg.STATS):
        os.makedirs(cfg.STATS, exist_ok=True)

    # initialize dataloader
    train_loader, val_loader = build_dataloader(cfg)
    # initialize model
    model = build_model(cfg)

    # The result folder is cfg.OUTPUT_PATH
    # The model save folder is cfg.MODEL_SAVE_PATH
    model_save_path = cfg.MODEL_SAVE_PATH
    if cfg.LOG_FILE_NAME is not None and len(cfg.LOG_FILE_NAME) > 0:
        logger_suffix = cfg.LOG_FILE_NAME
    else:
        logger_suffix = NOW_TIME
    logger_name = f"{cfg.MODEL_NAME}_{logger_suffix}"
    logger_id = None

    callbacks = []

    logger = WandbLogger(
        project=cfg.PROJECT,
        name=logger_name,
        id=logger_id,
        save_dir=model_save_path,
    )

    all_gpus = list(cfg.GPUS)
    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu",
        devices=all_gpus,
        strategy=None,
        callbacks=callbacks,
    )

    # automatically detect existing checkpoints
    ckp_files = os.listdir(model_save_path)
    ckp_files = [ckp for ckp in ckp_files if "model_" in ckp]
    if cfg.WEIGHT_FILE:  # if specify a weight file, load it
        # check if it has training states, or just a model weight
        ckp = torch.load(cfg.WEIGHT_FILE, map_location="cpu")
        # if it has, then it's a checkpoint compatible with pl
        if "state_dict" in ckp.keys():
            ckp_path = cfg.WEIGHT_FILE
        # if it's just a weight, then manually load it to the model
        else:
            ckp_path = None
            model.load_state_dict(ckp, strict=False)
    elif ckp_files:  # if not specify a weight file, we will check it for you
        ckp_files = sorted(
            ckp_files,
            key=lambda x: os.path.getmtime(os.path.join(model_save_path, x)),
        )
        last_ckp = ckp_files[-1]
        print(f"INFO: automatically detect checkpoint {last_ckp}")
        ckp_path = os.path.join(model_save_path, last_ckp)
    else:
        ckp_path = None

    model = JointSegmentationAlignmentModel.load_from_checkpoint(checkpoint_path=ckp_path, strict=False)
    print("Finish Setting -----")
    trainer.test(model, val_loader)

    print("Done training")


if __name__ == "__main__":
    from utils.config import cfg
    from utils.parse_args import parse_args
    from utils.print_easydict import print_easydict
    from utils.dup_stdout_manager import DupStdoutFileManager

    args = parse_args("Jigsaw")

    torch.manual_seed(cfg.RANDOM_SEED)

    parallel_strategy = "dp" if len(cfg.GPUS) > 1 else ""
    if len(cfg.GPUS) > 1:
        cfg.BATCH_SIZE *= len(cfg.GPUS)
        cfg.NUM_WORKERS *= len(cfg.GPUS)

    file_suffix = NOW_TIME
    if cfg.LOG_FILE_NAME is not None and len(cfg.LOG_FILE_NAME) > 0:
        file_suffix += "_{}".format(cfg.LOG_FILE_NAME)
    full_log_name = f"eval_log_{file_suffix}"

    with DupStdoutFileManager(
            os.path.join(cfg.OUTPUT_PATH, f"{full_log_name}.log")
    ) as _:
        print_easydict(cfg)

        test_model(cfg)
