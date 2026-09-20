import json
import os
import random
from dataclasses import dataclass, field
from functools import partial

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torch.utils.data.sampler import Sampler, BatchSampler
import itertools
import random

from ..utils.system_utils.config import parse_structured, instantiate_from_config
from ..utils.typing import *
from .utils import safe_dataloader

@dataclass
class DataModuleConfig:
    dataset: Dict[str, Any] = field(default_factory=dict)

    batch_size: int = 1
    eval_batch_size: int = 1

    num_workers: int = 16

    val_dataset: Dict[str, Any] = field(default_factory=dict)
    test_dataset: Dict[str, Any] = field(default_factory=dict)

    batch_sampler: bool = False
    shuffle_train: bool = False

    split: str = ""


class UniDataModule(pl.LightningDataModule):
    cfg: DataModuleConfig

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.cfg = parse_structured(DataModuleConfig, kwargs)


    def setup(self, stage=None) -> None:
        train_datasets, val_datasets, test_datasets = [], [], []

        if stage in [None, "fit"]:
            for dataset_name in self.cfg.dataset:
                data_cfg = self.cfg.dataset[dataset_name]
                split = self.cfg.split if self.cfg.split != "" else "train"
                train_datasets.append(instantiate_from_config(data_cfg, split=split))
            self.train_collate_fn = train_datasets[0].collate

        if stage in [None, "fit", "validate"]:
            for dataset_name in self.cfg.val_dataset:
                data_cfg = self.cfg.val_dataset[dataset_name]
                split = self.cfg.split if self.cfg.split != "" else "test"
                val_datasets.append(instantiate_from_config(data_cfg, split=split))
            self.val_collate_fn = val_datasets[0].collate

        if stage in [None, "test", "predict"]:
            test_datasets_cfg = (
                self.cfg.test_dataset
                if len(self.cfg.test_dataset) > 0
                else self.cfg.val_dataset
            )
            for dataset_name in test_datasets_cfg:
                data_cfg = test_datasets_cfg[dataset_name]
                split = self.cfg.split if self.cfg.split != "" else "test"
                test_datasets.append(instantiate_from_config(data_cfg, split=split))
            self.test_collate_fn = test_datasets[0].collate

        # Initialize datasets, ensuring they exist even if empty
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        if len(train_datasets) > 0:
            self.train_dataset = ConcatDataset(train_datasets)
        if len(val_datasets) > 0:
            self.val_dataset = ConcatDataset(val_datasets)
        if len(test_datasets) > 0:
            self.test_dataset = ConcatDataset(test_datasets)

    def prepare_data(self):
        pass
    
    @safe_dataloader("train_dataset")
    def train_dataloader(self) -> DataLoader:
        if not self.cfg.batch_sampler:
            return DataLoader(
                self.train_dataset,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_workers,
                collate_fn=self.train_collate_fn,
                shuffle=self.cfg.shuffle_train,
            )
        
        from miraccm.data.utils import SingleDatasetBatchSampler

        # Create DistributedSampler (if using DDP)
        distributed_sampler = None
        if hasattr(self.trainer, "world_size") and self.trainer.world_size > 1:
            from torch.utils.data.distributed import DistributedSampler

            distributed_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=self.cfg.shuffle_train,
            )

        train_sampler = SingleDatasetBatchSampler(
            self.train_dataset,
            self.cfg.batch_size,
            shuffle=self.cfg.shuffle_train,
            drop_last=False,
            sampler=distributed_sampler,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=train_sampler,
            num_workers=self.cfg.num_workers,
            collate_fn=self.train_collate_fn,
        )
    
    @safe_dataloader("val_dataset")
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.val_collate_fn,
            shuffle=False,
        )
    
    @safe_dataloader("test_dataset")
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.test_collate_fn,
            shuffle=False,
        )
    
    @safe_dataloader("test_dataset")
    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()