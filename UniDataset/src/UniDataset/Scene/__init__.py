"""Dataset classes exposed for configuration-based construction.

Imports are lazy so lightweight consumers do not need optional rendering/OpenCV
packages unless the corresponding dataset is instantiated.
"""

_EXPORTS = {
    "BlenderProcSceneDepthDataset": (".threedfrontv4.blenderproc_scene_depth", "BlenderProcSceneDepthDataset"),
    "BlenderProcSceneDepthDatasetConfig": (".threedfrontv4.blenderproc_scene_depth", "BlenderProcSceneDepthDatasetConfig"),
    "InfinigenCompositeSceneDepthDataset": (".infinigen_composite_scene_depth", "InfinigenCompositeSceneDepthDataset"),
    "InfinigenCompositeSceneDepthDatasetConfig": (".infinigen_composite_scene_depth", "InfinigenCompositeSceneDepthDatasetConfig"),
    "ObjaverseSceneDepthDataset": (".objaverse_scene_depth_dataset", "ObjaverseSceneDepthDataset"),
    "ObjaverseSceneDepthAlphaDataset": (".objaverse_scene_depth_dataset_alpha", "ObjaverseSceneDepthAlphaDataset"),
    "EvalDataset": (".eval_dataset", "EvalDataset"),
}
__all__ = list(_EXPORTS)

def __getattr__(name):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    from importlib import import_module
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
