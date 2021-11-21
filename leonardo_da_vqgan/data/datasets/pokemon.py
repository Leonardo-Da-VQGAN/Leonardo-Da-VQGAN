import importlib
from typing import Optional, Tuple
import PIL
from PIL import Image
from rich import print, pretty, inspect, traceback
pretty.install()
traceback.install()
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import torch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F


class PokemonDataset(Dataset):
    def __init__(self, 
            root: str, 
            transforms = None, 
            vis_threshold: float = 0.25,
            batch_size: int = 6,
            train = True,
            val = False
        ):

        self.root = root
        self.transforms = transforms
        self._vis_threshold = vis_threshold
        self._batch_size = batch_size
        self.train = train
        self.val = val
        set_names = ['img_id']
        if self.train:
            if self.val:
                self.set = pd.read_csv(f"{root}/gen3and4shinies/gt.txt", usecols=range(0,len(set_names)), names=set_names, header=None)
            else:
                self.set = pd.read_csv(f"{root}/gen3and4sprites/gt.txt", usecols=range(0,len(set_names)), names=set_names, header=None)
        else:
            self.set = pd.read_csv(f"{root}/test_set/gt.txt", usecols=range(0,len(set_names)), names=set_names, header=None)

        self.len = self.set.shape[0]

    def __str__(self) -> str:
        return f"{self.root}"
 
    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.train:
            if self.val:
                img_path = f"{self.root}/gen3and4shinies/{self.set['img_id'].iloc[idx]}"
            else:
                img_path = f"{self.root}/gen3and4sprites/{self.set['img_id'].iloc[idx]}"
        else:
            img_path = f"{self.root}/test_set/{self.set['img_id'].iloc[idx]}"


        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
            img = F.interpolate(img.unsqueeze(0), (80))
            img = img.squeeze(0)

        return img

    def __len__(self) -> int:
        return self.len

class Pokemon(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dims = (3, 80, 80)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    # def prepare_data(self):
    #     PokemonDataset(self.data_dir, train=True, download=True)
    #     PokemonDataset(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.dataset_train = PokemonDataset(self.data_dir, batch_size=self.batch_size,
                                   transforms=self.transform, train=True, val=False)
            self.dataset_val = PokemonDataset(self.data_dir, batch_size=self.batch_size,
                                   transforms=self.transform, train=True, val=True)

        if stage == "test" or stage is None:
            self.dataset_test = PokemonDataset(self.data_dir, batch_size=self.batch_size,
                                   transforms=self.transform, train=False)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)


    def __len__(self) -> int:
        return self.len

    @property
    def num_classes(self) -> int:
        """[summary]

        Returns:
            int: [description]
        """
        return len(self.classes)
