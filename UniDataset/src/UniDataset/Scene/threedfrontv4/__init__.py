"""ThreeDFront dataset implementations used by Mira-CCM."""

__all__ = ["BlenderProcSceneDepthDataset", "BlenderProcSceneDepthDatasetConfig"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from .blenderproc_scene_depth import (
        BlenderProcSceneDepthDataset,
        BlenderProcSceneDepthDatasetConfig,
    )
    value = locals()[name]
    globals()[name] = value
    return value
