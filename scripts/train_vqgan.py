import os, argparse, datetime
import time
import importlib
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

from leonardo_da_vqgan.models.vqgan import VQModel

from leonardo_da_vqgan.data.datasets.pokemon import Pokemon
from leonardo_da_vqgan.utils.utils import instantiate_from_config, mkdir

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

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

def train(config_path: str = os.getcwd()+"/config/custom_vqgan.yaml", job: str = "model"):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # seed
    pl.seed_everything(int(time.time()))
    np.random.seed(int(time.time()))
    torch.manual_seed(int(time.time()))
    torch.cuda.manual_seed(int(time.time()))
    # config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(device)
    # args
    print(config_path)
    
    # define the argument parse
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    
    # get arguments
    parsed_args, unknown = parser.parse_known_args()
    configs = [OmegaConf.load(cfg) for cfg in parsed_args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    # set up model name
    if parsed_args.name:
        name = "_"+parsed_args.name
    elif parsed_args.base:
        cfg_fname = os.path.split(parsed_args.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = "_"+cfg_name
    else:
        name = ""
    nowname = now+name+parsed_args.postfix

    # set up directories
    logdir = os.path.join("logs", nowname)
    mkdir(logdir)
    ckptdir = os.path.join(logdir, "checkpoints")
    mkdir(ckptdir)
    cfgdir = os.path.join(logdir, "configs")
    mkdir(ckptdir)
    
    # set up overall and trainer configs
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    for k in nondefault_trainer_args(parsed_args):
        trainer_config[k] = getattr(parsed_args, k)
    
    # confirm gpu count
    gpuinfo = trainer_config["gpus"]
    cpu = False
    print(f"Running on GPUs {gpuinfo}")
   
    # get trainer args
    trainer_args = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # instantiate vqgan model
    vqmodel = instantiate_from_config(config.model)
    
    trainer_kwargs = dict()
    # logger config setup
    default_logger_cfgs = {
        "wandb": {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "name": nowname,
                "save_dir": logdir,
                "offline": parsed_args.debug,
                "id": nowname,
            }
        },
    }
    
    # get logger configs
    default_logger_cfg = default_logger_cfgs["wandb"]
    logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # define default checkpoint config
    default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
    
    if hasattr(vqmodel, "monitor"):
        print(f"Monitoring {vqmodel.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = vqmodel.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 3
    # instantiate checkpoint configs
    modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

    # define data paths
    data_path = config['DATA_PATH']
    output_path = config["OUTPUT_PATH"]
    project = config["PROJECT"]

    # hyperparamters
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
    
    # data
    dataset = Pokemon(data_path, batch_size=batch_size, num_workers=num_workers)
    # dataset.prepare_data()
    dataset.setup()

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "leonardo_da_vqgan.utils.utils.SetupCallback",
            "params": {
                "resume": parsed_args.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            }
        },
        "image_logger": {
            "target": "leonardo_da_vqgan.utils.utils.ImageLogger",
            "params": {
                "batch_frequency": 50,
                "max_images": 12,
                "clamp": True
            }
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
            }
        },
    }
    # instantiate callback configs
    callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    # instantiate trainer
    trainer = Trainer.from_argparse_args(trainer_args, **trainer_kwargs)
    trainer.log_every_n_steps = 12

    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    if not cpu:
        ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
    else:
        ngpu = 1
    accumulate_grad_batches = 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    vqmodel.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
    print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
        vqmodel.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
   
    # vqmodel.eval().requires_grad_(False)
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    if parsed_args.train:
        try:
            trainer.fit(vqmodel, dataset)
        except Exception:
            melk()
            raise
    if not parsed_args.no_test and not trainer.interrupted:
        trainer.test(vqmodel, dataset)
    

if __name__ == '__main__':
    fire.Fire(train)
