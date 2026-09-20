from functools import partial, wraps
from typing import Iterator, List

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Sampler, SequentialSampler
from torch.utils.data.sampler import BatchSampler


def safe_dataloader(dataset_attr_name):
    """
    Decorator to safely handle dataloader methods.
    When the corresponding dataset does not exist, return a virtual dataloader.

    Args:
        dataset_attr_name (str): dataset attribute name (e.g. 'train_dataset', 'val_dataset', 'test_dataset')
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if the dataset exists
            if (
                not hasattr(self, dataset_attr_name)
                or getattr(self, dataset_attr_name) is None
            ):
                # Return a virtual dataloader, for DeepSpeed etc.
                from torch.utils.data import TensorDataset

                dummy_dataset = TensorDataset(torch.tensor([0]))
                return DataLoader(dummy_dataset, batch_size=1)

            # If the dataset exists, call the original method
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class SingleDatasetBatchSampler(BatchSampler):
    """
    Simple BatchSampler that ensures each batch contains samples from only one dataset.
    Supports both single-GPU and multi-GPU training.

    - When shuffle=False: traverse datasets sequentially (dataset1 -> dataset2 -> ...).
    - When shuffle=True: still keeps each batch within a single sub-dataset, but shuffles
      the *order of batches across sub-datasets*, so batches are interleaved randomly.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        sampler: Sampler = None,  # For DDP
    ):
        # Create a dummy sampler for BatchSampler inheritance (we override __iter__/__len__)
        if sampler is None:
            sampler = SequentialSampler(dataset)
        super().__init__(sampler, batch_size, drop_last)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.distributed_sampler = sampler

        self.dataset_lengths = [len(d) for d in self.dataset.datasets]

    def __iter__(self) -> Iterator[List[int]]:
        # Check if this is distributed training
        if hasattr(self.distributed_sampler, "num_replicas") and hasattr(
            self.distributed_sampler, "rank"
        ):
            num_replicas = self.distributed_sampler.num_replicas
            rank = self.distributed_sampler.rank
        else:
            num_replicas = 1
            rank = 0

        # 1) Build per-sub-dataset batches (each batch strictly from one sub-dataset)
        all_batches: List[List[int]] = []
        for dataset_idx, dataset_length in enumerate(self.dataset_lengths):
            if dataset_length <= 0:
                continue

            # Generate indices for current sub-dataset
            if self.shuffle:
                local_indices = torch.randperm(dataset_length).tolist()
            else:
                local_indices = list(range(dataset_length))

            # If distributed training, keep only the portion for current rank
            if num_replicas > 1:
                local_indices = [
                    idx
                    for i, idx in enumerate(local_indices)
                    if i % num_replicas == rank
                ]

            # Convert to global indices within ConcatDataset
            offset = (
                self.dataset.cumulative_sizes[dataset_idx - 1] if dataset_idx > 0 else 0
            )
            global_indices = [idx + offset for idx in local_indices]

            # Group by batch_size to generate batches for this sub-dataset
            for i in range(0, len(global_indices), self.batch_size):
                batch = global_indices[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)

        # 2) Shuffle batch order across sub-datasets to interleave them randomly
        if self.shuffle and len(all_batches) > 1:
            perm = torch.randperm(len(all_batches)).tolist()
            all_batches = [all_batches[i] for i in perm]

        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        # Check if this is distributed training
        if hasattr(self.distributed_sampler, "num_replicas") and hasattr(
            self.distributed_sampler, "rank"
        ):
            num_replicas = self.distributed_sampler.num_replicas
            rank = self.distributed_sampler.rank
        else:
            num_replicas = 1
            rank = 0

        total_batches = 0
        for dataset_length in self.dataset_lengths:
            # Calculate number of samples assigned to current rank
            if num_replicas > 1:
                rank_samples = dataset_length // num_replicas
                if rank < dataset_length % num_replicas:
                    rank_samples += 1
            else:
                rank_samples = dataset_length

            # Calculate number of batches
            if self.drop_last:
                batches = rank_samples // self.batch_size
            else:
                batches = (rank_samples + self.batch_size - 1) // self.batch_size

            total_batches += batches

        return total_batches
