from typing import List, Tuple
import os
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import hashlib
import requests
from tqdm import tqdm

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}

def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)

def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path

def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()

class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs = val_samples

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        logits = pl_module(val_imgs)[0]
        preds = torch.argmax(logits, 1)
        trainer.logger[1].experiment.log({
            "examples": [wandb.Image(x, caption=f"Pred:{pred}") 
                            for x, pred in zip(val_imgs, preds)],
            "global_step": trainer.global_step
            })

def collate_fn(batch: List[torch.Tensor]) -> Tuple[Tuple[torch.Tensor]]:
    """[summary]
    Args:
        batch (List[torch.Tensor]): [description]
    Returns:
        Tuple[Tuple[torch.Tensor]]: [description]
    """
    return tuple(zip(*batch))

def warmup_lr_scheduler(optimizer: torch.optim, warmup_iters: int,
                        warmup_factor: float) -> torch.optim:
    """[summary]
    Args:
        optimizer (torch.optim): [description]
        warmup_iters (int): [description]
        warmup_factor (float): [description]
    Returns:
        torch.optim: [description]
    """
    def fun(iter_num: int) -> float:
        if iter_num >= warmup_iters:
            return 1
        alpha = float(iter_num) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fun)

def mkdir(path: str):
    """[summary]
    Args:
        path (str): [description]
    """
    try:
        os.makedirs(path)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise