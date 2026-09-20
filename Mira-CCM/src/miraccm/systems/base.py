import inspect
import os
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ..utils.system_utils.base import (
    Updateable,
    update_end_if_possible,
    update_if_possible,
)
from ..utils.system_utils.config import parse_structured
from ..utils.system_utils.logging import debug, info, warn
from ..utils.system_utils.misc import (
    C,
    cleanup,
    get_device,
    get_rank,
    get_local_rank,
    load_module_weights,
    show_vram_usage,
)
from ..utils.system_utils.optimizer import parse_optimizer, parse_scheduler
from ..utils.system_utils.saving import SaverMixin
from ..utils.typing import *


class BaseSystem(pl.LightningModule, Updateable, SaverMixin):
    @dataclass
    class Config:
        optimizer: dict = field(default_factory=dict)
        scheduler: Optional[dict] = None
        weights: Optional[str] = None
        weights_ignore_modules: Optional[List[str]] = None
        weights_mapping: Optional[List[Dict[str, str]]] = None
        check_train_every_n_steps: int = 0
        check_val_limit_rank: int = 8
        cleanup_after_validation_step: bool = False
        cleanup_after_test_step: bool = False
        allow_tf32: bool = True

    cfg: Config

    def __init__(self, resumed=False, **kwargs) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, kwargs)
        self._non_modules = {}
        self._save_dir: Optional[str] = None
        self._resumed: bool = resumed
        self._resumed_eval: bool = False
        self._resumed_eval_status: dict = {"global_step": 0, "current_epoch": 0}

        # weird fix for extra VRAM usage on rank 0
        # credit: https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
        # torch.cuda.set_device(get_rank())
        torch.cuda.set_device(get_local_rank())
        print("local rank:", get_local_rank())
        print("rank:", get_rank())
        torch.cuda.empty_cache()

        torch.backends.cuda.matmul.allow_tf32 = self.cfg.allow_tf32

        self.configure()
        if self.cfg.weights is not None:
            self.load_weights(
                self.cfg.weights,
                self.cfg.weights_ignore_modules,
                self.cfg.weights_mapping,
            )
        self.post_configure()

    def register_non_module(self, name: str, module: torch.nn.Module) -> None:
        # non-modules won't be treated as model parameters
        self._non_modules[name] = module

    def non_module(self, name: str):
        return self._non_modules[name]

    def debug_unused_parameters(self, batch):
        from collections import defaultdict

        all_params = {
            name: param
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        info(f"Find all trainable parameters: {len(all_params)}")

        loss = self(batch)
        if isinstance(loss, dict):
            loss = loss["loss"]
        loss.backward(retain_graph=True)

        used_params = {}
        unused_params = {}

        for name, param in all_params.items():
            if param.grad is not None and torch.any(param.grad != 0):
                used_params[name] = param
            else:
                unused_params[name] = param

        info(f"Used parameters: {len(used_params)}")
        warn(f"Unused parameters: {len(unused_params)}")

        if unused_params:
            print("Unused parameters: ")
            unused_by_module = defaultdict(list)
            for param_name in unused_params.keys():
                module_name = (
                    param_name.split(".")[0] if "." in param_name else param_name
                )
                unused_by_module[module_name].append(param_name)

            for module_name, param_names in unused_by_module.items():
                print(f"  {module_name}: {len(param_names)} parameters")
                for param_name in param_names:
                    print(f"    - {param_name}")

        # Clear gradients
        self.zero_grad()

        return loss

    def load_weights(
        self,
        weights: str,
        ignore_modules: Optional[List[str]] = None,
        mapping: Optional[List[Dict[str, str]]] = None,
    ):
        state_dict, epoch, global_step = load_module_weights(
            weights,
            ignore_modules=ignore_modules,
            mapping=mapping,
            map_location="cpu",
        )
        self.load_state_dict(state_dict, strict=False)
        # restore step-dependent states
        self.do_update_step(epoch, global_step, on_load_weights=True)

    def set_resume_status(self, current_epoch: int, global_step: int):
        # restore correct epoch and global step in eval
        self._resumed_eval = True
        self._resumed_eval_status["current_epoch"] = current_epoch
        self._resumed_eval_status["global_step"] = global_step

    @property
    def resumed(self):
        # whether from resumed checkpoint
        return self._resumed

    @property
    def true_global_step(self):
        if self._resumed_eval:
            return self._resumed_eval_status["global_step"]
        else:
            return self.global_step

    @property
    def true_current_epoch(self):
        if self._resumed_eval:
            return self._resumed_eval_status["current_epoch"]
        else:
            return self.current_epoch

    def configure(self) -> None:
        pass

    def post_configure(self) -> None:
        """
        executed after weights are loaded
        """
        pass

    def C(self, value: Any) -> float:
        return C(value, self.true_current_epoch, self.true_global_step)

    def configure_optimizers(self):
        optim = parse_optimizer(self.cfg.optimizer, self)
        ret = {
            "optimizer": optim,
        }
        if self.cfg.scheduler is not None:
            ret.update(
                {
                    "lr_scheduler": parse_scheduler(self.cfg.scheduler, optim),
                }
            )
        return ret

    def on_fit_start(self) -> None:
        if self._save_dir is not None:
            info(f"Validation results will be saved to {self._save_dir}")
        else:
            warn(
                f"Saving directory not set for the system, visualization results will not be saved"
            )

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def check_train(self, batch, batch_idx=0, **kwargs):
        if (
            self.global_rank == 0
            and self.cfg.check_train_every_n_steps > 0
            and self.true_global_step % self.cfg.check_train_every_n_steps == 0
        ):
            # Check if the on_check_train method has the required parameters
            sig = inspect.signature(self.on_check_train)
            params = sig.parameters

            all_args = {"batch": batch, "batch_idx": batch_idx, **kwargs}

            # Filter out the parameters that are not in the on_check_train method
            filtered_args = {}
            for param_name in params.keys():
                if param_name in all_args:
                    filtered_args[param_name] = all_args[param_name]

            self.on_check_train(**filtered_args)

    def on_check_train(self, batch, outputs, **kwargs):
        pass

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_test_epoch_end(self):
        pass

    def on_test_end(self) -> None:
        if self._save_dir is not None:
            info(f"Test results saved to {self._save_dir}")

    def on_predict_start(self) -> None:
        pass

    def predict_step(self, batch, batch_idx):
        pass

    def on_predict_epoch_end(self) -> None:
        pass

    def on_predict_end(self) -> None:
        pass

    def preprocess_data(self, batch, stage):
        pass

    """
    Implementing on_after_batch_transfer of DataModule does the same.
    But on_after_batch_transfer does not support DP.
    """

    def on_train_batch_start(self, batch, batch_idx, unused=0):
        self.preprocess_data(batch, "train")
        self.dataset = self.trainer.train_dataloader.dataset
        update_if_possible(self.dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.preprocess_data(batch, "validation")
        self.dataset = self.trainer.val_dataloaders.dataset
        update_if_possible(self.dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.preprocess_data(batch, "test")
        self.dataset = self.trainer.test_dataloaders.dataset
        update_if_possible(self.dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.preprocess_data(batch, "predict")
        self.dataset = self.trainer.predict_dataloaders.dataset
        update_if_possible(self.dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.dataset = self.trainer.train_dataloader.dataset
        update_end_if_possible(
            self.dataset, self.true_current_epoch, self.true_global_step
        )
        self.do_update_step_end(self.true_current_epoch, self.true_global_step)

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.dataset = self.trainer.val_dataloaders.dataset
        update_end_if_possible(
            self.dataset, self.true_current_epoch, self.true_global_step
        )
        self.do_update_step_end(self.true_current_epoch, self.true_global_step)
        if self.cfg.cleanup_after_validation_step:
            # cleanup to save vram
            cleanup()

    def on_test_batch_end(self, outputs, batch, batch_idx):
        self.dataset = self.trainer.test_dataloaders.dataset
        update_end_if_possible(
            self.dataset, self.true_current_epoch, self.true_global_step
        )
        self.do_update_step_end(self.true_current_epoch, self.true_global_step)
        if self.cfg.cleanup_after_test_step:
            # cleanup to save vram
            cleanup()

    def on_predict_batch_end(self, outputs, batch, batch_idx):
        self.dataset = self.trainer.predict_dataloaders.dataset
        update_end_if_possible(
            self.dataset, self.true_current_epoch, self.true_global_step
        )
        self.do_update_step_end(self.true_current_epoch, self.true_global_step)
        if self.cfg.cleanup_after_test_step:
            # cleanup to save vram
            cleanup()

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        pass

    def on_before_optimizer_step(self, optimizer):
        """
        # some gradient-related debugging goes here, example:
        from lightning.pytorch.utilities import grad_norm
        norms = grad_norm(self.geometry, norm_type=2)
        print(norms)
        for name, p in self.named_parameters():
            if p.grad is None:
                info(f"{name} does not receive gradients!")
        """
        pass
