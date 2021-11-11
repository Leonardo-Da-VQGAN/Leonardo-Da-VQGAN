import os, argparse, datetime
import random
import string
import yaml
import fire
from rich import print, pretty, inspect, traceback
pretty.install()
traceback.install()
import torch
import numpy as np
import pytorch_lightning as pl
import os
import wandb
from pytorch_lightning import loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import OmegaConf
from leonardo_da_vqgan.utils import utils

from leonardo_da_vqgan.models.vqgan import VQModel, instantiate_from_config

from leonardo_da_vqgan.data.datasets.pokemon import Pokemon

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )

    return parser

def train(config_path: str = os.getcwd()+"/config/custom_vqgan.yaml", job: str = "model"):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # seed
    pl.seed_everything(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(device)
    # args
    print(config_path)
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()
    
    if opt.name:
        name = "_"+opt.name
    elif opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = "_"+cfg_name
    else:
        name = ""
    nowname = now+name+opt.postfix
    logdir = os.path.join("logs", nowname)

    default_logger_cfgs = {
        "wandb": {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "name": nowname,
                "save_dir": logdir,
                "offline": opt.debug,
                "id": nowname,
            }
        },
    }
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    trainer_config = lightning_config.get("trainer", OmegaConf.create())

    data_path = config['DATA_PATH']
    output_path = config["OUTPUT_PATH"]
    project = config["PROJECT"]

    # hypterparamters
    batch_size = int(config["data"]['params']['batch_size'])
    num_workers = int(config['data']['params']['num_workers'])
    learning_rate = float(config['model']["base_learning_rate"])
    max_epoch = int(config['data']['params']['max_epoch'])
    gpus = 1
    print(f"BatchSize:{batch_size} - Workers:{num_workers} - LR:{learning_rate} - MaxEpochs:{max_epoch}")
    # https://github.com/tchaton/lightning-geometric/blob/master/examples/utils/loggers.py
    # output directory
    random_str = ''.join(random.choices(
        string.ascii_uppercase + string.digits, k=5))
    experiment = f"{random_str}_{job}_{batch_size}_{learning_rate}_{max_epoch}"
    output_dir = f"{output_path}/checkpoints/{experiment}"
    logger = f"{output_path}/logs/{experiment}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # loggers
    wandb.login()
    tb_logger = loggers.TensorBoardLogger(save_dir=logger)
    wandb_logger = loggers.WandbLogger(
        project=project, log_model="all", name=experiment)
    
    # data
    dataset = Pokemon(data_path, batch_size=batch_size, num_workers=num_workers)
    # dataset.prepare_data()
    dataset.setup()

    val_images = next(iter(dataset.val_dataloader()))

    # model
    vqmodel = instantiate_from_config(config['model'])
    # vqmodel.eval().requires_grad_(False)
    
    wandb_logger.watch(vqmodel)
    _loggers = [tb_logger, wandb_logger]
    _callbacks = [ModelCheckpoint(dirpath=output_dir), utils.ImagePredictionLogger(val_images),LearningRateMonitor(logging_interval='step')]
    
    # trainer
    trainer = pl.Trainer(
        logger=_loggers,
        callbacks=_callbacks,
        gpus=gpus,
        max_epochs=max_epoch,
        progress_bar_refresh_rate=20
    )
    trainer.fit(vqmodel, dataset)


if __name__ == '__main__':
    fire.Fire(train)
