import numpy as np
import trimesh
from typing import Any, Optional, Union, Tuple, List


def normalize_scene(
    meshes: List[trimesh.Trimesh],
    local2world_transforms: List[np.ndarray],
    scene_bounds: np.ndarray,
    min_bound: float = -0.5,
    max_bound: float = 0.5,
    align_ground: bool = False,
    return_scale: bool = False
) -> Union[Tuple[List[np.ndarray], trimesh.Scene], Tuple[List[np.ndarray], trimesh.Scene, float]]:
    """
    将组合的场景归一化到指定的bounds之间
    
    Args:
        meshes: trimesh.Trimesh对象的列表
        local2world_transforms: 对应的local2world变换矩阵列表，每个为4x4矩阵
        scene_bounds: 场景的边界框，shape为(2, 3)，[min_point, max_point]
        min_bound: 归一化后的最小边界，默认为-0.5
        max_bound: 归一化后的最大边界，默认为0.5
        align_ground: 是否将地面对齐到min_bound平面（y轴）。
                     如果为True，z轴居中，y轴地面对齐到min_bound， x轴不平移；
                     如果为False，所有轴都居中到目标范围中心
        return_scale: 是否返回缩放因子。如果为True，返回三元组(transforms, scene, scale)；
                     如果为False，返回二元组(transforms, scene)，保持向后兼容
        
    Returns:
        如果return_scale=False（默认）:
            (normalized_transforms, scene): 
                - normalized_transforms: 归一化后的local2world变换矩阵列表
                - scene: 包含归一化后的meshes的trimesh.Scene对象
        如果return_scale=True:
            (normalized_transforms, scene, scale): 
                - normalized_transforms: 归一化后的local2world变换矩阵列表
                - scene: 包含归一化后的meshes的trimesh.Scene对象
                - scale: 归一化的缩放因子
            
    Raises:
        ValueError: 如果meshes列表为空或meshes和transforms长度不匹配
    """
    if not meshes:
        raise ValueError("meshes列表不能为空")
    
    if len(meshes) != len(local2world_transforms):
        raise ValueError("meshes和local2world_transforms长度必须相同")
    
    # 使用传入的scene_bounds
    min_point = scene_bounds[0]
    max_point = scene_bounds[1]
    
    # 计算场景的中心和大小
    center = (min_point + max_point) / 2.0
    size = max_point - min_point
    max_size = np.max(size)
    
    # 计算缩放因子（使得最大维度从max_size缩放到max_bound - min_bound）
    scale = (max_bound - min_bound) / max_size
    
    if align_ground:
        # 地面对齐模式：z轴居中，y轴地面对齐到min_bound，x轴不平移
        translation = np.zeros(3)
        # translation[0] = -scale * center[0]  # x轴保持中心位置
        translation[1] = min_bound - scale * min_point[1]  # y轴将地面移到min_bound
        translation[2] = -scale * center[2]  # z轴保持中心位置
    else:
        # 默认模式：所有轴都居中
        target_center = (min_bound + max_bound) / 2.0
        translation = target_center - scale * center
    
    # 创建归一化的变换矩阵
    # 先缩放，再平移
    T_norm = np.eye(4)
    T_norm[:3, :3] = scale * np.eye(3)
    T_norm[:3, 3] = translation
    
    # 应用归一化变换到所有的local2world_transforms
    normalized_transforms = []
    for transform in local2world_transforms:
        normalized_transform = T_norm @ transform
        normalized_transforms.append(normalized_transform)
    
    # 创建归一化后的scene
    normalized_scene = trimesh.Scene()
    for mesh, normalized_transform in zip(meshes, normalized_transforms):
        normalized_scene.add_geometry(mesh, transform=normalized_transform)
    
    if return_scale:
        return normalized_transforms, normalized_scene, scale
    else:
        return normalized_transforms, normalized_scene


def normalize_scene_transforms(
    local2world_transforms: List[np.ndarray],
    scene_bounds: np.ndarray,
    min_bound: float = -0.5,
    max_bound: float = 0.5,
    max_scale: float = 1000.0,
    scale: Optional[float] = None,
    align_ground: bool = False,
    return_scale: bool = False
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], float]]:
    """
    计算场景归一化的变换矩阵，不需要meshes输入，也不构建scene
    
    Args:
        local2world_transforms: 对应的local2world变换矩阵列表，每个为4x4矩阵
        scene_bounds: 场景的边界框，shape为(2, 3)，[min_point, max_point]
        min_bound: 归一化后的最小边界，默认为-0.5
        max_bound: 归一化后的最大边界，默认为0.5
        max_scale: 最大缩放因子
        align_ground: 是否将地面对齐到min_bound平面（y轴）。
                     如果为True，z轴居中，y轴地面对齐到min_bound， x轴不平移；
                     如果为False，所有轴都居中到目标范围中心
        return_scale: 是否返回缩放因子
        
    Returns:
        如果return_scale=False（默认）:
            normalized_transforms: 归一化后的local2world变换矩阵列表
        如果return_scale=True:
            (normalized_transforms, scale): 归一化后的变换矩阵列表和缩放因子
    """
    if len(local2world_transforms) == 0:
        raise ValueError("local2world_transforms列表不能为空")
    
    # 使用传入的scene_bounds
    min_point = scene_bounds[0]
    max_point = scene_bounds[1]
    
    # 计算场景的中心和大小
    center = (min_point + max_point) / 2.0
    size = max_point - min_point
    max_size = np.max(size)
    
    # 计算缩放因子（使得最大维度从max_size缩放到max_bound - min_bound）
    scale = (max_bound - min_bound) / max_size
    scale = min(scale, max_scale)  # 限制最大缩放因子
    
    if align_ground:
        # 地面对齐模式：z轴居中，y轴地面对齐到min_bound，x轴不平移
        translation = np.zeros(3)
        # translation[0] = -scale * center[0]  # x轴不平移
        translation[1] = min_bound - scale * min_point[1]  # y轴将地面移到min_bound
        translation[2] = -scale * center[2]  # z轴保持中心位置
    else:
        # 默认模式：所有轴都居中
        target_center = (min_bound + max_bound) / 2.0
        translation = target_center - scale * center
    
    # 创建归一化的变换矩阵
    # 先缩放，再平移
    T_norm = np.eye(4)
    T_norm[:3, :3] = scale * np.eye(3)
    T_norm[:3, 3] = translation
    
    # 应用归一化变换到所有的local2world_transforms
    normalized_transforms = []
    for transform in local2world_transforms:
        normalized_transform = T_norm @ transform
        normalized_transforms.append(normalized_transform)
    
    if return_scale:
        return normalized_transforms, scale
    else:
        return normalized_transforms


def normalize_object(trimesh_obj: trimesh.Trimesh, margin: float = 0.02) -> Tuple[trimesh.Trimesh, np.ndarray]:
    # Normalize the object to fit within (-1, -1, -1) and (1, 1, 1) with a margin.
    bounds = trimesh_obj.bounds
    obj_min, obj_max = bounds
    obj_center = (obj_min + obj_max) / 2.0
    extents = obj_max - obj_min
    max_extent = extents.max()

    # Define a margin (e.g., 2% margin from each side)
    target_half_size = 0.5 - margin

    # Compute scale factor to fit the longest side of the object into the target cube
    scale_factor = target_half_size * 2 / max_extent

    # Build transformation: first translate to origin, then scale.
    # To center and scale: v_new = scale * (v_old - center) = scale * v_old - scale * center
    # So we need a transform that scales by scale_factor and translates by -scale_factor * center
    
    # Method 1: compose separate matrices
    translation_matrix = trimesh.transformations.translation_matrix(-obj_center)
    scaling_matrix = trimesh.transformations.scale_matrix(scale_factor)
    normalize_transform = trimesh.transformations.concatenate_matrices(scaling_matrix, translation_matrix)

    trimesh_obj.apply_transform(normalize_transform)

    return trimesh_obj, normalize_transform



def align_horizon(T):
    # 给定transformation，得到对齐矫正矩阵：使物体底面与xz平面平行
    # 旋转部分
    R = T[:3, :3]

    # local 底面法向 (y=0 平面)
    n_local = np.array([0.0, 1.0, 0.0])

    # 变换到 world
    n_world = R @ n_local
    n_world = n_world / (np.linalg.norm(n_world) + 1e-8)

    # 世界 up 方向
    up = np.array([0.0, 1.0, 0.0])

    cos_theta = np.dot(n_world, up)

    # 如果已经平行，直接跳过
    if cos_theta < 0.999:

        # 如果朝下，翻转
        if cos_theta < 0:
            up = -up
            cos_theta = np.dot(n_world, up)

        # 旋转轴
        axis = np.cross(n_world, up)
        axis_norm = np.linalg.norm(axis)

        if axis_norm > 1e-8:

            axis = axis / axis_norm
            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

            # Rodrigues 公式
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])

            I = np.eye(3)

            R_align = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

            # 构造 4x4
            T_align = np.eye(4)
            T_align[:3, :3] = R_align
    else:
        T_align = np.eye(4)

    return T_align