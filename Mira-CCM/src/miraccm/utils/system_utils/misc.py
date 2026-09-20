import gc
import importlib
import os
import re
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field

import psutil
import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig, OmegaConf
from packaging import version
from torch.distributed import get_rank, get_world_size
from torch.distributed.nn.functional import all_gather

from ..typing import *
from .config import config_to_primitive


### Find class
def find(cls_string):
    module_string = ".".join(cls_string.split(".")[:-1])
    cls_name = cls_string.split(".")[-1]
    module = importlib.import_module(module_string, package=None)
    cls = getattr(module, cls_name)
    return cls


### Version management
def parse_version(ver: str):
    return version.parse(ver)


### Distributed training utilities
def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0

def get_local_rank():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return 0

def get_device():
    return torch.device(f"cuda:{get_rank()}")


### Model checkpoint and weights management
def load_module_weights(
    path, module_name=None, ignore_modules=None, mapping=None, map_location=None
) -> Tuple[dict, int, int]:
    if module_name is not None and ignore_modules is not None:
        raise ValueError("module_name and ignore_modules cannot be both set")
    if map_location is None:
        map_location = get_device()

    ckpt = torch.load(path, map_location=map_location)
    state_dict = ckpt["state_dict"]

    if mapping is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            if any([k.startswith(m["to"]) for m in mapping]):
                pass
            else:
                state_dict_to_load[k] = v
        for k, v in state_dict.items():
            for m in mapping:
                if k.startswith(m["from"]):
                    k_dest = k.replace(m["from"], m["to"])
                    info(f"Mapping {k} => {k_dest}")
                    state_dict_to_load[k_dest] = v.clone()
        state_dict = state_dict_to_load

    state_dict_to_load = state_dict

    if ignore_modules is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            ignore = any(
                [k.startswith(ignore_module + ".") for ignore_module in ignore_modules]
            )
            if ignore:
                continue
            state_dict_to_load[k] = v

    if module_name is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            m = re.match(rf"^{module_name}\.(.*)$", k)
            if m is None:
                continue
            state_dict_to_load[m.group(1)] = v

    return state_dict_to_load, ckpt["epoch"], ckpt["global_step"]


### Configuration and value scaling
def C(value: Any, epoch: int, global_step: int) -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = config_to_primitive(value)
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        if isinstance(end_step, int):
            current_step = global_step
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
        elif isinstance(end_step, float):
            current_step = epoch
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
    return value


### Memory management utilities
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    try:
        import tinycudann as tcnn

        tcnn.free_temporary_memory()
    except:
        pass


def finish_with_cleanup(func: Callable):
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        cleanup()
        return out

    return wrapper


### Distributed communication utilities
def _distributed_available():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def barrier():
    if not _distributed_available():
        return
    else:
        torch.distributed.barrier()


def broadcast(tensor, src=0):
    if not _distributed_available():
        return tensor
    else:
        torch.distributed.broadcast(tensor, src=src)
        return tensor


### Model parameter utilities
def enable_gradient(model, enabled: bool = True) -> None:
    for param in model.parameters():
        param.requires_grad_(enabled)


### Time and performance measurement
class TimeRecorder:
    _instance = None

    def __init__(self):
        self.items = {}
        self.accumulations = defaultdict(list)
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"
        self.enabled = False

    def __new__(cls):
        # singleton
        if cls._instance is None:
            cls._instance = super(TimeRecorder, cls).__new__(cls)
        return cls._instance

    def enable(self, enabled: bool) -> None:
        self.enabled = enabled

    def start(self, name: str) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize()
        self.items[name] = time.time()

    def end(self, name: str, accumulate: bool = False) -> float:
        if not self.enabled or name not in self.items:
            return
        torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        if accumulate:
            self.accumulations[name].append(delta)
        t = delta * self.time_scale
        info(f"{name}: {t:.2f}{self.time_unit}")

    def get_accumulation(self, name: str, average: bool = False) -> float:
        if not self.enabled or name not in self.accumulations:
            return
        acc = self.accumulations.pop(name)
        total = sum(acc)
        if average:
            t = total / len(acc) * self.time_scale
        else:
            t = total * self.time_scale
        info(f"{name} for {len(acc)} times: {t:.2f}{self.time_unit}")


### Global time recorder instance
time_recorder = TimeRecorder()


@contextmanager
def time_recorder_enabled():
    enabled = time_recorder.enabled
    time_recorder.enable(enabled=True)
    try:
        yield
    finally:
        time_recorder.enable(enabled=enabled)


### Memory usage monitoring
def show_vram_usage(name):
    available, total = torch.cuda.mem_get_info()
    used = total - available
    print(
        f"{name}: {used / 1024**2:.1f}MB, {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.1f}MB"
    )


### Set trainable modules
def set_trainable_modules(
    model,
    trainable_names: List[str],
    non_trainable_names: Optional[List[str]] = None,
):
    def is_module_trainable(name):
        trainable = any([n in name for n in trainable_names])
        if non_trainable_names is not None:
            non_trainable = any([n in name for n in non_trainable_names])
            return trainable and not non_trainable
        else:
            return trainable

    if trainable_names and len(trainable_names) > 0:
        model.requires_grad_(False)
        for name, module in model.named_modules():
            if is_module_trainable(name):
                module.requires_grad_(True)
            else:
                module.requires_grad_(False)
    else:
        model.requires_grad_(True)

    return model


# Use this if your model has nn.Parameter
def set_trainable_parameters(
    model, trainable_names: List[str], non_trainable_names: Optional[List[str]] = None
):
    def is_parameter_trainable(name):
        trainable = any([n in name for n in trainable_names])
        if non_trainable_names is not None:
            non_trainable = any([n in name for n in non_trainable_names])
            return trainable and not non_trainable
        else:
            return trainable

    if trainable_names and len(trainable_names) > 0:
        for name, param in model.named_parameters():
            if is_parameter_trainable(name):
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            param.requires_grad = True

    return model


### Distributed utilities
def all_gather_batch(tensor):
    """Gather tensor from all GPUs and concatenate along batch dimension."""
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor

    chunks = all_gather(tensor)
    return torch.cat(chunks, dim=0)
