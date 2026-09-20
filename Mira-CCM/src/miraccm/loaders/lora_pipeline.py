"""
The mixin is adapted to support different versions of diffusers.
The following features are supported in diffusers >= 0.32.0
- transformer.load_lora_adapter
- _fetch_state_dict in diffusers.loaders.lora_base
"""

import inspect
import os
from typing import Callable, Dict, List, Optional, Union

import torch
from diffusers.loaders.lora_base import LoraBaseMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    get_adapter_name,
    get_peft_kwargs,
    is_peft_available,
    is_peft_version,
    is_torch_version,
    is_transformers_available,
    is_transformers_version,
    logging,
)
from huggingface_hub.utils import validate_hf_hub_args

_LOW_CPU_MEM_USAGE_DEFAULT_LORA = False
if is_torch_version(">=", "1.9.0"):
    if (
        is_peft_available()
        and is_peft_version(">=", "0.13.1")
        and is_transformers_available()
        and is_transformers_version(">", "4.45.2")
    ):
        _LOW_CPU_MEM_USAGE_DEFAULT_LORA = True

logger = logging.get_logger(__name__)

TRANSFORMER_NAME = "transformer"
IMAGE_ENCODER_NAME = "image_encoder"


class ImgToShapeLoraLoaderMixin(LoraBaseMixin):
    r"""
    LoRA Loader Mixin for [`VecsetHunyuanDiTModel`] and [`Dinov2Model`], designed for [`Hunyuan3DPipeline`].
    """

    # Allow both 'transformer' and 'image_encoder' as LoRA-loadable modules
    _lora_loadable_modules = ["transformer", "image_encoder"]
    transformer_name = TRANSFORMER_NAME
    image_encoder_name = IMAGE_ENCODER_NAME

    @classmethod
    @validate_hf_hub_args
    def lora_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        **kwargs,
    ):
        r"""
        Return the state dict and metadata for LoRA weights of the transformer and image_encoder.

        Args:
            pretrained_model_name_or_path_or_dict (`str`, `os.PathLike`, or `dict`):
                - HuggingFace Hub model id
                - Local directory containing model weights
                - Directly provided state dict
            Other arguments are consistent with the diffusers LoRA system.
        """
        # Parse common arguments
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        return_lora_metadata = kwargs.pop("return_lora_metadata", False)

        # Prefer safetensors, fallback to pickle if needed
        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        user_agent = {"file_type": "attn_procs_weights", "framework": "pytorch"}

        # NOTE: to support different versions of diffusers
        try:
            from diffusers.loaders.lora_base import _fetch_state_dict

            # Use diffusers utility to fetch the state dict
            state_dict, metadata = _fetch_state_dict(
                pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict,
                weight_name=weight_name,
                use_safetensors=use_safetensors,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
                allow_pickle=allow_pickle,
            )
        except:  # NOTE: for older version of diffusers
            state_dict = cls._fetch_state_dict(
                pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict,
                weight_name=weight_name,
                use_safetensors=use_safetensors,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
                allow_pickle=allow_pickle,
            )
            metadata = None

        # Filter out DoRA weights (not supported by diffusers currently)
        is_dora_scale_present = any("dora_scale" in k for k in state_dict)
        if is_dora_scale_present:
            warn_msg = "Detected DoRA weights, which are not supported by diffusers. Filtering out related parameters. Please report if this is an issue."
            logger.warning(warn_msg)
            state_dict = {k: v for k, v in state_dict.items() if "dora_scale" not in k}

        # Return state dict and metadata
        out = (state_dict, metadata) if return_lora_metadata else state_dict
        return out

    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[str] = None,
        hotswap: bool = False,
        **kwargs,
    ):
        """
        Load LoRA weights into `self.transformer` and `self.image_encoder`.
        All kwargs are forwarded to lora_state_dict.
        """
        # Check PEFT backend
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required.")

        # Whether to use low CPU memory loading
        low_cpu_mem_usage = kwargs.pop(
            "low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT_LORA
        )
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` requires peft>=0.13.0. Please upgrade peft."
            )

        # Avoid in-place modification of dict
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = (
                pretrained_model_name_or_path_or_dict.copy()
            )

        # Fetch LoRA weights and metadata
        kwargs["return_lora_metadata"] = True
        state_dict, metadata = self.lora_state_dict(
            pretrained_model_name_or_path_or_dict, **kwargs
        )

        # Check weight format
        is_correct_format = all("lora" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint format.")

        # Load weights into transformer
        self.load_lora_into_transformer(
            state_dict,
            transformer=(
                getattr(self, self.transformer_name)
                if not hasattr(self, "transformer")
                else self.transformer
            ),
            adapter_name=adapter_name,
            metadata=metadata,
            _pipeline=self,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
        )

        # Load weights into image_encoder (only if image_encoder LoRA weights exist)
        image_encoder_keys = [
            k for k in state_dict.keys() if k.startswith(self.image_encoder_name)
        ]
        if image_encoder_keys:
            self.load_lora_into_image_encoder(
                state_dict,
                image_encoder=(
                    getattr(self, self.image_encoder_name)
                    if not hasattr(self, "image_encoder")
                    else self.image_encoder
                ),
                adapter_name=adapter_name,
                metadata=metadata,
                _pipeline=self,
                low_cpu_mem_usage=low_cpu_mem_usage,
                hotswap=hotswap,
            )
        else:
            logger.info(
                "No image_encoder LoRA weights found in state_dict, skipping image_encoder LoRA loading."
            )

    # NOTE: for older version of diffusers
    @classmethod
    def load_lora_into_transformer_old(
        cls,
        state_dict,
        transformer,
        adapter_name=None,
        _pipeline=None,
        low_cpu_mem_usage=False,
    ):
        """
        This will load the LoRA layers specified in `state_dict` into `transformer`.

        Parameters:
            state_dict (`dict`):
                A standard state dict containing the lora layer parameters. The keys can either be indexed directly
                into the unet or prefixed with an additional `unet` which can be used to distinguish between text
                encoder lora layers.
            transformer (`SD3Transformer2DModel`):
                The Transformer model to load the LoRA layers into.
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
            Speed up model loading by only loading the pretrained LoRA weights and not initializing the random weights.:
        """
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

        keys = list(state_dict.keys())

        transformer_keys = [k for k in keys if k.startswith(cls.transformer_name)]
        state_dict = {
            k.replace(f"{cls.transformer_name}.", ""): v
            for k, v in state_dict.items()
            if k in transformer_keys
        }

        if len(state_dict.keys()) > 0:
            # check with first key if is not in peft format
            first_key = next(iter(state_dict.keys()))
            if "lora_A" not in first_key:
                try:
                    from diffusers.loaders.lora_base import (
                        convert_unet_state_dict_to_peft,
                    )

                    state_dict = convert_unet_state_dict_to_peft(state_dict)
                except ImportError:
                    # For older versions of diffusers
                    logger.warning(
                        "convert_unet_state_dict_to_peft not available, skipping conversion"
                    )

            if adapter_name in getattr(transformer, "peft_config", {}):
                raise ValueError(
                    f"Adapter name {adapter_name} already in use in the transformer - please select a new adapter name."
                )

            rank = {}
            for key, val in state_dict.items():
                if "lora_B" in key:
                    rank[key] = val.shape[1]

            lora_config_kwargs = get_peft_kwargs(
                rank, network_alpha_dict=None, peft_state_dict=state_dict
            )
            if "use_dora" in lora_config_kwargs:
                if lora_config_kwargs["use_dora"] and is_peft_version("<", "0.9.0"):
                    raise ValueError(
                        "You need `peft` 0.9.0 at least to use DoRA-enabled LoRAs. Please upgrade your installation of `peft`."
                    )
                else:
                    lora_config_kwargs.pop("use_dora")
            lora_config = LoraConfig(**lora_config_kwargs)

            # adapter_name
            if adapter_name is None:
                adapter_name = get_adapter_name(transformer)

            # In case the pipeline has been already offloaded to CPU - temporarily remove the hooks
            # otherwise loading LoRA weights will lead to an error
            is_model_cpu_offload, is_sequential_cpu_offload = (
                cls._optionally_disable_offloading(_pipeline)
            )

            peft_kwargs = {}
            if is_peft_version(">=", "0.13.1"):
                peft_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

            inject_adapter_in_model(
                lora_config, transformer, adapter_name=adapter_name, **peft_kwargs
            )
            incompatible_keys = set_peft_model_state_dict(
                transformer, state_dict, adapter_name, **peft_kwargs
            )

            warn_msg = ""
            if incompatible_keys is not None:
                # Check only for unexpected keys.
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    lora_unexpected_keys = [
                        k for k in unexpected_keys if "lora_" in k and adapter_name in k
                    ]
                    if lora_unexpected_keys:
                        warn_msg = (
                            f"Loading adapter weights from state_dict led to unexpected keys found in the model:"
                            f" {', '.join(lora_unexpected_keys)}. "
                        )

                # Filter missing keys specific to the current adapter.
                missing_keys = getattr(incompatible_keys, "missing_keys", None)
                if missing_keys:
                    lora_missing_keys = [
                        k for k in missing_keys if "lora_" in k and adapter_name in k
                    ]
                    if lora_missing_keys:
                        warn_msg += (
                            f"Loading adapter weights from state_dict led to missing keys in the model:"
                            f" {', '.join(lora_missing_keys)}."
                        )

            if warn_msg:
                logger.warning(warn_msg)

            # Offload back.
            if is_model_cpu_offload:
                _pipeline.enable_model_cpu_offload()
            elif is_sequential_cpu_offload:
                _pipeline.enable_sequential_cpu_offload()
            # Unsafe code />

    @classmethod
    def load_lora_into_transformer(
        cls,
        state_dict,
        transformer,
        adapter_name=None,
        _pipeline=None,
        low_cpu_mem_usage=False,
        hotswap: bool = False,
        metadata=None,
    ):
        """
        Load LoRA weights into the transformer model.
        """
        # Check peft version
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` requires peft>=0.13.0. Please upgrade peft."
            )

        # Log info
        logger.info(f"Loading LoRA weights for {cls.transformer_name}.")
        # Actual loading
        try:
            transformer.load_lora_adapter(
                state_dict,
                network_alphas=None,
                adapter_name=adapter_name,
                metadata=metadata,
                _pipeline=_pipeline,
                low_cpu_mem_usage=low_cpu_mem_usage,
                hotswap=hotswap,
            )
        except:  # NOTE: for older version of diffusers
            cls.load_lora_into_transformer_old(
                state_dict, transformer, adapter_name, _pipeline, low_cpu_mem_usage
            )

    @classmethod
    def load_lora_into_image_encoder_old(
        cls,
        state_dict,
        image_encoder,
        adapter_name=None,
        _pipeline=None,
        low_cpu_mem_usage=False,
    ):
        """
        This will load the LoRA layers specified in `state_dict` into `image_encoder`.

        Parameters:
            state_dict (`dict`):
                A standard state dict containing the lora layer parameters.
            image_encoder (`Dinov2Model`):
                The Image Encoder model to load the LoRA layers into.
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model.
        """
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

        keys = list(state_dict.keys())

        image_encoder_keys = [k for k in keys if k.startswith(cls.image_encoder_name)]
        state_dict = {
            k.replace(f"{cls.image_encoder_name}.", ""): v
            for k, v in state_dict.items()
            if k in image_encoder_keys
        }

        if len(state_dict.keys()) > 0:
            # check with first key if is not in peft format
            first_key = next(iter(state_dict.keys()))
            if "lora_A" not in first_key:
                try:
                    from diffusers.loaders.lora_base import (
                        convert_unet_state_dict_to_peft,
                    )

                    state_dict = convert_unet_state_dict_to_peft(state_dict)
                except ImportError:
                    # For older versions of diffusers
                    logger.warning(
                        "convert_unet_state_dict_to_peft not available, skipping conversion"
                    )

            if adapter_name in getattr(image_encoder, "peft_config", {}):
                raise ValueError(
                    f"Adapter name {adapter_name} already in use in the image_encoder - please select a new adapter name."
                )

            rank = {}
            for key, val in state_dict.items():
                if "lora_B" in key:
                    rank[key] = val.shape[1]

            lora_config_kwargs = get_peft_kwargs(
                rank, network_alpha_dict=None, peft_state_dict=state_dict
            )
            if "use_dora" in lora_config_kwargs:
                if lora_config_kwargs["use_dora"] and is_peft_version("<", "0.9.0"):
                    raise ValueError(
                        "You need `peft` 0.9.0 at least to use DoRA-enabled LoRAs. Please upgrade your installation of `peft`."
                    )
                else:
                    lora_config_kwargs.pop("use_dora")
            lora_config = LoraConfig(**lora_config_kwargs)

            # adapter_name
            if adapter_name is None:
                adapter_name = get_adapter_name(image_encoder)

            # In case the pipeline has been already offloaded to CPU - temporarily remove the hooks
            is_model_cpu_offload, is_sequential_cpu_offload = (
                cls._optionally_disable_offloading(_pipeline)
            )

            peft_kwargs = {}
            if is_peft_version(">=", "0.13.1"):
                peft_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

            inject_adapter_in_model(
                lora_config, image_encoder, adapter_name=adapter_name, **peft_kwargs
            )
            incompatible_keys = set_peft_model_state_dict(
                image_encoder, state_dict, adapter_name, **peft_kwargs
            )

            warn_msg = ""
            if incompatible_keys is not None:
                # Check only for unexpected keys.
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    lora_unexpected_keys = [
                        k for k in unexpected_keys if "lora_" in k and adapter_name in k
                    ]
                    if lora_unexpected_keys:
                        warn_msg = (
                            f"Loading adapter weights from state_dict led to unexpected keys found in the model:"
                            f" {', '.join(lora_unexpected_keys)}. "
                        )

                # Filter missing keys specific to the current adapter.
                missing_keys = getattr(incompatible_keys, "missing_keys", None)
                if missing_keys:
                    lora_missing_keys = [
                        k for k in missing_keys if "lora_" in k and adapter_name in k
                    ]
                    if lora_missing_keys:
                        warn_msg += (
                            f"Loading adapter weights from state_dict led to missing keys in the model:"
                            f" {', '.join(lora_missing_keys)}."
                        )

            if warn_msg:
                logger.warning(warn_msg)

            # Offload back.
            if is_model_cpu_offload:
                _pipeline.enable_model_cpu_offload()
            elif is_sequential_cpu_offload:
                _pipeline.enable_sequential_cpu_offload()

    @classmethod
    def load_lora_into_image_encoder(
        cls,
        state_dict,
        image_encoder,
        adapter_name=None,
        _pipeline=None,
        low_cpu_mem_usage=False,
        hotswap: bool = False,
        metadata=None,
    ):
        """
        Load LoRA weights into the image encoder model.
        """
        # Check peft version
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` requires peft>=0.13.0. Please upgrade peft."
            )

        # Log info
        logger.info(f"Loading LoRA weights for {cls.image_encoder_name}.")
        # Actual loading
        try:
            image_encoder.load_lora_adapter(
                state_dict,
                network_alphas=None,
                adapter_name=adapter_name,
                metadata=metadata,
                _pipeline=_pipeline,
                low_cpu_mem_usage=low_cpu_mem_usage,
                hotswap=hotswap,
            )
        except:  # NOTE: for older version of diffusers
            cls.load_lora_into_image_encoder_old(
                state_dict, image_encoder, adapter_name, _pipeline, low_cpu_mem_usage
            )

    @classmethod
    def save_lora_weights(
        cls,
        save_directory: Union[str, os.PathLike],
        transformer_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        image_encoder_lora_layers: Dict[
            str, Union[torch.nn.Module, torch.Tensor]
        ] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
        transformer_lora_adapter_metadata: Optional[dict] = None,
        image_encoder_lora_adapter_metadata: Optional[dict] = None,
    ):
        r"""
        Save LoRA weights for the transformer and image_encoder.
        """
        state_dict = {}
        lora_adapter_metadata = {}

        # Check input
        if not transformer_lora_layers and not image_encoder_lora_layers:
            raise ValueError(
                "You must pass at least one of `transformer_lora_layers` or `image_encoder_lora_layers`."
            )

        # Pack transformer weights
        if transformer_lora_layers:
            state_dict.update(
                cls.pack_weights(transformer_lora_layers, cls.transformer_name)
            )

        # Pack image_encoder weights
        if image_encoder_lora_layers:
            state_dict.update(
                cls.pack_weights(image_encoder_lora_layers, cls.image_encoder_name)
            )

        # Pack metadata
        if transformer_lora_adapter_metadata is not None:
            from diffusers.loaders.lora_base import _pack_dict_with_prefix

            lora_adapter_metadata.update(
                _pack_dict_with_prefix(
                    transformer_lora_adapter_metadata, cls.transformer_name
                )
            )

        if image_encoder_lora_adapter_metadata is not None:
            from diffusers.loaders.lora_base import _pack_dict_with_prefix

            lora_adapter_metadata.update(
                _pack_dict_with_prefix(
                    image_encoder_lora_adapter_metadata, cls.image_encoder_name
                )
            )

        # Prepare arguments for write_lora_layers
        write_kwargs = {
            "state_dict": state_dict,
            "save_directory": save_directory,
            "is_main_process": is_main_process,
            "weight_name": weight_name,
            "save_function": save_function,
            "safe_serialization": safe_serialization,
        }

        # Check if write_lora_layers accepts lora_adapter_metadata parameter
        write_lora_layers_sig = inspect.signature(cls.write_lora_layers)
        if "lora_adapter_metadata" in write_lora_layers_sig.parameters:
            write_kwargs["lora_adapter_metadata"] = lora_adapter_metadata
        else:
            # NOTE: for older version of diffusers
            # Log warning if metadata is provided but not supported
            if lora_adapter_metadata:
                logger.warning(
                    "lora_adapter_metadata provided but write_lora_layers does not support it. "
                    "Metadata will be ignored."
                )

        # Save to disk
        cls.write_lora_layers(**write_kwargs)

    def fuse_lora(
        self,
        components: List[str] = ["transformer", "image_encoder"],
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
        adapter_names: Optional[List[str]] = None,
        **kwargs,
    ):
        r"""
        Fuse LoRA parameters into the original transformer and image_encoder weights (experimental API).
        """
        # Call parent method
        super().fuse_lora(
            components=components,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
            adapter_names=adapter_names,
            **kwargs,
        )

    def unfuse_lora(
        self, components: List[str] = ["transformer", "image_encoder"], **kwargs
    ):
        r"""
        Undo the effect of fuse_lora (experimental API).
        """
        # Call parent method
        super().unfuse_lora(components=components, **kwargs)
