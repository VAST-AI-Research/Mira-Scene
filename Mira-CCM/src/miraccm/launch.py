"""

Features:
- Multi-GPU training support with automatic device management
- Colored logging output for better readability
- Automatic checkpoint and code snapshot management
- Integration with TensorBoard and Weights & Biases logging
- Configurable callbacks and loggers
- Benchmark mode for performance measurement
- Dynamic type checking support

Usage:
    # Training mode
    python launch.py --config configs/train.yaml --train --gpu 0,1

    # Validation mode
    python launch.py --config configs/validate.yaml --validate --gpu 0

    # Testing mode
    python launch.py --config configs/test.yaml --test --gpu 0

    # Export mode
    python launch.py --config configs/export.yaml --export --gpu 0

    # With Weights & Biases logging
    python launch.py --config configs/train.yaml --train --wandb

    # With verbose logging and benchmarking
    python launch.py --config configs/train.yaml --train --verbose --benchmark

Arguments:
    --config: Path to YAML configuration file (required)
    --gpu: GPU(s) to use (default: "0")
           - "0": Use first GPU
           - "0,1": Use first and second GPUs
           - If CUDA_VISIBLE_DEVICES is set, this argument is ignored
    --train: Enable training mode
    --validate: Enable validation mode
    --test: Enable testing mode
    --export: Enable export mode
    --wandb: Enable Weights & Biases logging
    --verbose: Set logging level to DEBUG
    --benchmark: Enable benchmark mode for timing measurements
    --typecheck: Enable dynamic type checking

Default Configurations:
    - Accelerator: "gpu"
    - Devices: All available GPUs (or filtered by CUDA_VISIBLE_DEVICES)
    - Inference mode: False
    - Default callbacks (training mode):
        * ModelCheckpoint: Saves model checkpoints
        * LearningRateMonitor: Monitors learning rate
        * CodeSnapshotCallback: Saves code snapshots
        * ConfigSnapshotCallback: Saves configuration snapshots
        * CustomProgressBar: Custom progress bar
    - Default loggers (training mode):
        * TensorBoardLogger: TensorBoard logging
        * WandbLogger: Weights & Biases logging (if --wandb is specified)
    - Checkpoint directory: {trial_dir}/ckpts
    - TensorBoard logs: {trial_dir}/tb_logs
    - Code snapshots: {trial_dir}/code
    - Config snapshots: {trial_dir}/configs

Environment Variables:
    - CUDA_VISIBLE_DEVICES: Controls which GPUs are visible to the process
    - CUDA_DEVICE_ORDER: Set to "PCI_BUS_ID" for consistent GPU ordering
"""

import argparse
import contextlib
import logging
import os
import sys


def main(args, extras) -> None:
    # set CUDA_VISIBLE_DEVICES if needed, then import pytorch-lightning
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0]

    # Always rely on CUDA_VISIBLE_DEVICES if specific GPU ID(s) are specified.
    # As far as Pytorch Lightning is concerned, we always use all available GPUs
    # (possibly filtered by CUDA_VISIBLE_DEVICES).
    devices = -1
    if len(env_gpus) > 0:
        # CUDA_VISIBLE_DEVICES was set already, e.g. within SLURM srun or higher-level script.
        n_gpus = len(env_gpus)
    else:
        selected_gpus = list(args.gpu.split(","))
        n_gpus = len(selected_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import pytorch_lightning as pl
    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
    from torch.serialization import add_safe_globals

    # If the checkpoint is trusted, you can force torch.load to use weights_only=False
    # to avoid repeatedly adding safe_globals. Lightning will call torch.load internally.
    _torch_load = torch.load

    def _torch_load_force_full(*args, **kwargs):
        kwargs["weights_only"] = False
        return _torch_load(*args, **kwargs)

    torch.load = _torch_load_force_full

    if args.typecheck:
        from jaxtyping import install_import_hook

        install_import_hook("miraccm", "typeguard.typechecked")

    from .systems.base import BaseSystem
    from .utils.system_utils.callbacks import (
        CodeSnapshotCallback,
        ConfigSnapshotCallback,
        CustomProgressBar,
        ProgressCallback,
    )
    from .utils.system_utils.config import (
        ExperimentConfig,
        instantiate_from_config,
        load_config,
    )
    from .utils.system_utils.misc import get_rank, time_recorder
    from .utils.typing import Optional

    logger = logging.getLogger("pytorch_lightning")
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.benchmark:
        time_recorder.enable(True)

    for handler in logger.handlers:
        if handler.stream == sys.stderr:  # type: ignore
            from .utils.system_utils.logging import ColoredFilter

            handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
            handler.addFilter(ColoredFilter())

    # parse YAML config to OmegaConf
    cfg: ExperimentConfig
    cfg = load_config(args.config, cli_args=extras, n_gpus=n_gpus)

    # data
    dm: pl.LightningDataModule = instantiate_from_config(cfg.data)

    # system
    system: BaseSystem = instantiate_from_config(
        cfg.system, resumed=cfg.resume is not None
    )
    system.set_save_dir(os.path.join(cfg.trial_dir, "save"))

    callbacks = []
    # default callbacks for training
    if args.train:
        callbacks += [
            ModelCheckpoint(
                dirpath=os.path.join(cfg.trial_dir, "ckpts"), **cfg.checkpoint
            ),
            LearningRateMonitor(logging_interval="step"),
            CodeSnapshotCallback(
                os.path.join(cfg.trial_dir, "code"),
                code_root=os.path.dirname(os.path.abspath(__file__)),
                use_version=False
            ),
            ConfigSnapshotCallback(
                args.config,
                cfg,
                os.path.join(cfg.trial_dir, "configs"),
                use_version=False,
            ),
            CustomProgressBar(refresh_rate=1),
        ]
    # add custom callbacks
    if cfg.callbacks is not None:
        callbacks += instantiate_from_config(cfg.callbacks, recursive=True).values()

    def write_to_text(file, lines):
        with open(file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    loggers = []
    if cfg.loggers is not None:
        loggers += instantiate_from_config(cfg.loggers, recursive=True).values()
    else:
        # default callbacks for training
        if args.train:
            # make tensorboard logging dir to suppress warning
            rank_zero_only(
                lambda: os.makedirs(
                    os.path.join(cfg.trial_dir, "tb_logs"), exist_ok=True
                )
            )()
            loggers += [
                TensorBoardLogger(cfg.trial_dir, name="tb_logs"),
            ]
            if args.wandb:
                wandb_kwargs = dict(project=cfg.name, name=cfg.tag)
                if cfg.resume is not None:
                    # Resume wandb run if WANDB_RESUME_ID env var is set
                    wandb_resume_id = os.environ.get("WANDB_RESUME_ID", "")
                    if wandb_resume_id:
                        wandb_kwargs["id"] = wandb_resume_id
                        wandb_kwargs["resume"] = "must"
                wandb_logger = WandbLogger(**wandb_kwargs)
                system._wandb_logger = wandb_logger
                loggers += [wandb_logger]
            rank_zero_only(
                lambda: write_to_text(
                    os.path.join(cfg.trial_dir, "cmd.txt"),
                    ["python " + " ".join(sys.argv), str(args)],
                )
            )()

    trainer = Trainer(
        callbacks=callbacks,
        logger=loggers,
        inference_mode=False,
        accelerator="gpu",
        devices=devices,
        **cfg.trainer,
    )

    # set a different seed for each device
    # NOTE: use trainer.global_rank instead of get_rank() to avoid getting the local rank
    pl.seed_everything(cfg.seed + trainer.global_rank, workers=True)

    def set_system_status(system: BaseSystem, ckpt_path: Optional[str]):
        if ckpt_path is None:
            return
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        system.set_resume_status(ckpt["epoch"], ckpt["global_step"])

    if args.train:
        trainer.fit(system, datamodule=dm, ckpt_path=cfg.resume)
        trainer.test(system, datamodule=dm)
    elif args.validate:
        # manually set epoch and global_step as they cannot be automatically resumed
        set_system_status(system, cfg.resume)
        trainer.validate(system, datamodule=dm, ckpt_path=cfg.resume)
    elif args.test:
        # manually set epoch and global_step as they cannot be automatically resumed
        set_system_status(system, cfg.resume)
        trainer.test(system, datamodule=dm, ckpt_path=cfg.resume)
    elif args.export:
        set_system_status(system, cfg.resume)
        trainer.predict(system, datamodule=dm, ckpt_path=cfg.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
        "1,2 means use the 2nd and 3rd available GPU. "
        "If CUDA_VISIBLE_DEVICES is set before calling `launch.py`, "
        "this argument is ignored and all available GPUs are always used.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--export", action="store_true")

    parser.add_argument("--wandb", action="store_true", help="if true, log to wandb")

    parser.add_argument(
        "--verbose", action="store_true", help="if true, set logging level to DEBUG"
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="if true, set to benchmark mode to record running times",
    )

    parser.add_argument(
        "--typecheck",
        action="store_true",
        help="whether to enable dynamic type checking",
    )

    args, extras = parser.parse_known_args()

    main(args, extras)
