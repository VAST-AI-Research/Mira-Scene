
import importlib
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Any, Optional, Union


def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.merge(OmegaConf.structured(fields), cfg)
    return scfg


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, recursive: bool = False, **kwargs):
    """
    Instantiate config with nested `target` fields, supporting nested dicts and lists if recursive=True.
    If recursive=False, only instantiate the top-level target (do not recursively instantiate params).
    If recursive=True, kwargs will be ignored.
    """
    if not recursive:
        if "target" not in config:
            raise KeyError("Expected key `target` to instantiate.")

        cls = get_obj_from_str(config["target"])

        params = config.get("params", dict())
        kwargs.update(params)
        instance = cls(**kwargs)

        return instance

    if isinstance(config, (list, ListConfig)):
        # Recursively process each item in the list
        return [instantiate_from_config(item, recursive, **kwargs) for item in config]
    elif isinstance(config, (dict, DictConfig)):
        if "target" in config:
            # Recursively process the 'params' field if present
            params = config.get("params", dict())
            if isinstance(params, (dict, DictConfig)):
                params = {
                    k: instantiate_from_config(v, recursive) for k, v in params.items()
                }
            elif isinstance(params, (list, ListConfig)):
                params = [instantiate_from_config(v, recursive) for v in params]
            # Merge params with any additional kwargs
            merged_kwargs = {**params, **kwargs}
            cls = get_obj_from_str(config["target"])
            return cls(**merged_kwargs)
        else:
            # For a regular dict, recursively process each value
            return {k: instantiate_from_config(v, recursive) for k, v in config.items()}
    else:
        # For other types, return as is
        return config
