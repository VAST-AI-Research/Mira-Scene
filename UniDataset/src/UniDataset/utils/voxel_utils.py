import os
import open3d as o3d
import numpy as np
import torch
import trimesh
from typing import Any, Optional, Union, Tuple
import functools


def gen_voxel_grid(voxel_dict):
    voxel_indices = voxel_dict['voxel_indexes'].long()  # (1, N, 3)
    voxel_centers = voxel_dict['voxel_centers'].float()  # (1, N, 3)
    voxel_res = int(voxel_dict['voxel_res'])

    voxel = torch.zeros(voxel_res, voxel_res, voxel_res, dtype=torch.long)
    voxel[voxel_indices[:,0], voxel_indices[:,1], voxel_indices[:,2]] = 1
    return voxel



def voxel_cache(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        model_path = kwargs.get("model_path", None)
        save_dir = kwargs.get("save_dir", None)
        voxel_res = kwargs.get("voxel_res", None)
        cache_path = None

        if save_dir and model_path:
            # Map cache 1:1 to model_path (by model_id) without time/hash
            # Derive model_id from the parent folder name of the model file
            model_id = os.path.basename(os.path.dirname(model_path))
            cache_dir = os.path.join(save_dir, model_id)
            os.makedirs(cache_dir, exist_ok=True)
            cache_filename = f"voxel_{voxel_res}.pt" if voxel_res is not None else "voxel.pt"
            cache_path = os.path.join(cache_dir, cache_filename)
            if os.path.exists(cache_path):
                try:
                    # print("hit cache ... ", cache_path)
                    return torch.load(cache_path, map_location="cpu")
                except Exception:
                    pass

        voxel_dict = func(*args, **kwargs)

        if cache_path is not None:
            try:
                torch.save(voxel_dict, cache_path)
            except Exception:
                pass
        return voxel_dict

    return wrapper


@voxel_cache
def voxelize_trimesh_obj(
    trimesh_obj: trimesh.Trimesh,
    voxel_res: int = 64,
    model_path: Optional[str] = None,
    save_dir: Optional[str] = None,
):
    """Voxelize point cloud

    Args:
        trimesh_obj (trimesh.Trimesh): Trimesh object
        voxel_res (int): Voxel resolution
        model_path (str, optional): Path to source model for cache key
        save_dir (str, optional): Directory to save/load cached voxel

    Returns:
        voxel_dict (dict): Voxelization result
    """

    # Handle empty mesh
    if len(trimesh_obj.vertices) == 0 or len(trimesh_obj.faces) == 0:
        voxel_info = {
            'voxel_indexes': torch.zeros((0, 3), dtype=torch.long),
            'voxel_centers': torch.zeros((0, 3), dtype=torch.float32),
            'voxel_res': voxel_res,
        }
        return voxel_info

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(trimesh_obj.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(trimesh_obj.faces)

    voxel_size = 1.0 / voxel_res
    min_bound = np.array([-0.5, -0.5, -0.5])
    max_bound = np.array([0.5, 0.5, 0.5])

    # voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
    #     mesh, voxel_size=voxel_size
    # )
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh, voxel_size=voxel_size, min_bound=min_bound, max_bound=max_bound
    )

    voxels = voxel_grid.get_voxels()
    grid_index = np.stack([v.grid_index for v in voxels])
    origin = voxel_grid.origin
    
    # World centers
    voxel_centers_world = origin + (grid_index + 0.5) * voxel_size

    # Canonical indices
    voxel_indexes = np.floor((voxel_centers_world + 0.5) / voxel_size).astype(np.int32)
    
    # Filter valid
    valid = (voxel_indexes >= 0) & (voxel_indexes < voxel_res)
    valid = np.all(valid, axis=1)
    
    voxel_indexes = voxel_indexes[valid]
    
    # Unique
    voxel_indexes = np.unique(voxel_indexes, axis=0)

    # World centers
    voxel_centers = origin + (voxel_indexes.astype(np.float32) + 0.5) * voxel_size

    # to tensor
    voxel_indexes = torch.from_numpy(voxel_indexes).long()
    voxel_centers = torch.from_numpy(voxel_centers).float()

    voxel_info = {
        'voxel_indexes': voxel_indexes,
        'voxel_centers': voxel_centers,
        'voxel_res': voxel_res,
    }
    return voxel_info


def voxel_index_to_world_coord(voxel_index: torch.Tensor, voxel_res: int = 64, 
                                min_bound: float = -0.5, max_bound: float = 0.5) -> torch.Tensor:
    """
    将体素索引转换为世界坐标（体素中心）
    
    Args:
        voxel_index: [N, 3] int64 tensor, 体素索引
        voxel_res: 体素分辨率
        min_bound: 最小边界
        max_bound: 最大边界
    
    Returns:
        world_coord: [N, 3] float tensor, 世界坐标
    """
    voxel_size = (max_bound - min_bound) / voxel_res
    # 体素中心坐标 = min_bound + (index + 0.5) * voxel_size
    world_coord = min_bound + (voxel_index.float() + 0.5) * voxel_size
    return world_coord


def world_coord_to_voxel_index(world_coord: torch.Tensor, voxel_res: int = 64,
                                min_bound: float = -0.5, max_bound: float = 0.5) -> torch.Tensor:
    """
    将世界坐标转换为体素索引
    
    Args:
        world_coord: [N, 3] float tensor, 世界坐标
        voxel_res: 体素分辨率
        min_bound: 最小边界
        max_bound: 最大边界
    
    Returns:
        voxel_index: [N, 3] int64 tensor, 体素索引
    """
    voxel_size = (max_bound - min_bound) / voxel_res
    # index = floor((coord - min_bound) / voxel_size)
    voxel_index = torch.floor((world_coord - min_bound) / voxel_size).long()
    return voxel_index


def transform_and_merge_voxels(
    voxel_index_context: torch.Tensor,
    voxel_index_new: torch.Tensor,
    transform: dict,
    voxel_res: int = 64,
    min_bound: float = -0.5,
    max_bound: float = 0.5
) -> torch.Tensor:
    """
    将新的体素索引经过变换后合并到上下文体素中
    
    Args:
        voxel_index_context: [N1, 3] int64 tensor, 上下文体素索引（可能为空）
        voxel_index_new: [N2, 3] int64 tensor, 新的体素索引
        transform: dict包含:
            - 'R': [3, 3] 旋转矩阵
            - 't': [3] 平移向量
            - 's': scalar 缩放因子
        voxel_res: 体素分辨率，默认64
        min_bound: 最小边界，默认-0.5
        max_bound: 最大边界，默认0.5
    
    Returns:
        merged_voxel_index: [M, 3] int64 tensor, 合并后的体素索引
    """
    # 如果新体素为空，直接返回上下文
    if voxel_index_new.numel() == 0:
        return voxel_index_context
    
    # 提取变换参数
    R = transform['R']  # [3, 3]
    t = transform['t']  # [3]
    s = transform['s']  # scalar
    
    # 转换为tensor
    if not isinstance(R, torch.Tensor):
        R = torch.tensor(R, dtype=torch.float32, device=voxel_index_new.device)
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32, device=voxel_index_new.device)
    if not isinstance(s, torch.Tensor):
        s = torch.tensor(s, dtype=torch.float32, device=voxel_index_new.device)
    
    # 1. 将新的体素索引转换为世界坐标
    world_coords = voxel_index_to_world_coord(voxel_index_new, voxel_res, min_bound, max_bound)
    
    # 2. 应用变换: x' = s * R * x + t
    transformed_coords = (s * (world_coords @ R.T)) + t  # [N2, 3]
    
    # 3. 转换回体素索引
    transformed_voxel_index = world_coord_to_voxel_index(
        transformed_coords, voxel_res, min_bound, max_bound
    )
    
    # 4. 过滤超出边界的体素
    valid_mask = (
        (transformed_voxel_index >= 0).all(dim=1) & 
        (transformed_voxel_index < voxel_res).all(dim=1)
    )
    transformed_voxel_index = transformed_voxel_index[valid_mask]
    
    # 5. 合并上下文体素和变换后的体素
    if voxel_index_context.numel() == 0:
        merged_voxel_index = transformed_voxel_index
    else:
        # 拼接两个体素索引集合
        merged_voxel_index = torch.cat([voxel_index_context, transformed_voxel_index], dim=0)
    
    # 6. 去重：将索引转换为唯一标识符，然后去重
    if merged_voxel_index.numel() > 0:
        # 使用torch.unique去重
        merged_voxel_index = torch.unique(merged_voxel_index, dim=0)
    
    return merged_voxel_index


def gen_voxel_grid_simple(voxel_indexes, voxel_res=64):
    # voxel_indices: (N, 3), torch.Tensor, int64
    voxel = torch.zeros(voxel_res, voxel_res, voxel_res, dtype=torch.long).to(voxel_indexes.device)
    voxel[voxel_indexes[:,0], voxel_indexes[:,1], voxel_indexes[:,2]] = 1
    return voxel
