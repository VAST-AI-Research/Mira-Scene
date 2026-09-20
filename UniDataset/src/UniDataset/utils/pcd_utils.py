import open3d as o3d
import numpy as np
import json
import torch

def save_pcd(points, filename, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is None:
        colors = np.ones_like(points) * 0.5  # Default color is gray
    else:
        colors = np.clip(colors, 0, 1)  # Ensure colors are in [0, 1] range
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)  # Save the point cloud to a file


def read_blender_camera(camera_info_path, image_height=768, image_width=768):
    with open(camera_info_path, 'r') as f:
        camera_info = json.load(f)

    poses_list = []
    camera_angle_x_list = []
    camera_angle_x = float(camera_info["camera_angle_x"])
    for frame in camera_info["locations"]:
        poses_list.append(np.array(frame["transform_matrix"]))
        # camera_angle_x = np.deg2rad(camera_angle_x)
        camera_angle_x_list.append(camera_angle_x)

    # image_width, image_height = frame["width"], frame["height"]
    focal_length_list = [0.5 * image_width / np.tan(0.5 * camera_angle_x) for camera_angle_x in camera_angle_x_list]
    cx = image_width / 2.0
    cy = image_height / 2.0

    extrinsics = [np.linalg.inv(poses_list[i]) for i in range(len(poses_list))]  # default: c2w
    intrinsics = [np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]]) for focal_length in focal_length_list]

    return intrinsics, extrinsics


##### calculate pcd transformation, apply transformation
def compute_bounding_box_center(points):
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points)
    # points: (N, 3), torch.Tensor
    min_coords = points.min(dim=0)[0]  # (3,)
    max_coords = points.max(dim=0)[0]  # (3,)
    center = (min_coords + max_coords) / 2
    return center

def transform_pcd(points, param=None, inverse=False):
    # points: (N, 3), torch.Tensor
    if param is None:
        # calculate param from points to [-1, 1] cube
        center = compute_bounding_box_center(points)[None]  # (1,3)
        ##### instead of eulidean distance, use max distance along each axis
        # radius = ((points - center) ** 2).sum(dim=-1).sqrt()  # (N,)
        radius = torch.max(torch.abs(points - center), dim=-1)[0]  # (N,)
        radius = torch.quantile(radius, 0.999) * 1.02
        param = {
            "scale": 1.0 / radius.item(),
            "translation": -center * (1.0 / radius.item())
        }

    if inverse:
        return (points - param["translation"]) / param["scale"], param  # (N, 3)
    else:
        return points * param["scale"] + param["translation"], param  # (N, 3)


##### voxelize point cloud
def voxelize_pcd(points, voxel_res=64):
    # points: (N, 3), torch.Tensor, range [-1, 1]
    voxel_size = 2.0 / voxel_res
    min_bound = np.array([-1.0, -1.0, -1.0])
    max_bound = np.array([1.0, 1.0, 1.0])

    pcd = o3d.geometry.PointCloud()
    points = torch.clamp(points, -1+1e-6, 1-1e-6)  # Ensure points are within [-1, 1]
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd, voxel_size, min_bound=min_bound, max_bound=max_bound
    )
    # voxel indices (i,j,k)
    indices = np.array([v.grid_index for v in voxel_grid.get_voxels()])
    indices = torch.as_tensor(indices, dtype=torch.long)

    # compute voxel centers
    # center = min_bound + (idx + 0.5) * voxel_size
    centers = min_bound + (indices.numpy() + 0.5) * voxel_size
    centers = torch.as_tensor(centers, dtype=torch.float32)  # (N,3)
    return indices, centers


def voxels_to_pcd(voxel, voxel_res=64, min_bound=-1.0, max_bound=1.0):
    indices = torch.nonzero(voxel > 0, as_tuple=False)  # (N, 3)
    voxel_size = (max_bound - min_bound) / voxel_res
    centers = min_bound + (indices.float() + 0.5) * voxel_size  # (N, 3)
    return centers  # torch.Tensor  

def voxels_indices_to_pcd(indices, voxel_res=64, min_bound=-1.0, max_bound=1.0):
    voxel_size = (max_bound - min_bound) / voxel_res
    centers = min_bound + (indices.float() + 0.5) * voxel_size  # (N, 3)
    return centers  # torch.Tensor  


def generate_uniform_voxel_centers(res, min_bound=-1.0, max_bound=1.0):
    voxel_size = (max_bound - min_bound) / res
    coords = torch.arange(res, dtype=torch.float32)

    # (res,res,res,3)
    grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing='ij')

    centers = torch.stack([
        min_bound + (grid_x + 0.5) * voxel_size,
        min_bound + (grid_y + 0.5) * voxel_size,
        min_bound + (grid_z + 0.5) * voxel_size,
    ], dim=-1)

    return centers   # shape (res, res, res, 3)


def sample_points_from_bbox(bbox, num_samples_per_dim=8):
    """
    Uniformly sample points from a bounding box.
    
    Args:
        bbox: [2, 3] array, [min_point, max_point] or [6] array [xmin, ymin, zmin, xmax, ymax, zmax]
        num_samples_per_dim: number of samples along each dimension (default 8 for 8x8x8=512 points)
    
    Returns:
        points: [num_samples_per_dim^3, 3] tensor of sampled points
    """
    if isinstance(bbox, np.ndarray):
        bbox = torch.from_numpy(bbox).float()
    
    min_point = bbox[0]
    max_point = bbox[1]
    
    # Generate uniform grid samples
    coords = torch.linspace(0, 1, num_samples_per_dim, dtype=torch.float32)
    grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing='ij')
    
    # Scale to bbox
    points = torch.stack([
        min_point[0] + grid_x * (max_point[0] - min_point[0]),
        min_point[1] + grid_y * (max_point[1] - min_point[1]),
        min_point[2] + grid_z * (max_point[2] - min_point[2]),
    ], dim=-1)
    
    return points.view(-1, 3)  # [num_samples_per_dim^3, 3]


def voxels_to_mesh(voxels, voxel_res=64):
    """
    Convert voxel integer coordinates into a triangle mesh.

    Args:
        voxels (torch.Tensor): Tensor of shape [N, 3] indicating voxel integer coordinates.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Vertices of shape [8*N, 3] and faces of shape [6*2*N, 3].
    """
    if isinstance(voxels, torch.Tensor):
        voxels = voxels.cpu().numpy()
    cube_vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ])
    cube_vertices = (cube_vertices) / (voxel_res / 2.0)  # Normalize to [-1, 1] range
    cube_faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 1, 5], [0, 5, 4],  # Front face
        [2, 3, 7], [2, 7, 6],  # Back face
        [1, 2, 6], [1, 6, 5],  # Right face
        [0, 3, 7], [0, 7, 4],  # Left face
    ])
    N = voxels.shape[0]
    voxel_vertices = ((voxels[:, None, :] / voxel_res * 2 - 1) + cube_vertices[None, :, :]).reshape(-1, 3)
    voxel_faces = (np.arange(N)[:, None, None] * 8 + cube_faces[None, :, :]).reshape(-1, 3)
    return voxel_vertices, voxel_faces


@torch.cuda.amp.autocast(enabled=False)
def compute_similarity_transform(src, tgt):
    """
    Compute the similarity transform (R, t, s) that best aligns src to tgt.
    Args:
        src: [N, 3] source points (normalized in [-1, 1])
        tgt: [N, 3] target points (scene space)
    Returns:
        R: [3, 3] rotation matrix
        t: [3] translation vector
        s: scalar scale factor
    """
    # Center the points
    src_mean = src.mean(0, keepdim=True)  # [1, 3]
    tgt_mean = tgt.mean(0, keepdim=True)  # [1, 3]
    src_centered = src - src_mean  # [N, 3]
    tgt_centered = tgt - tgt_mean  # [N, 3]

    # Compute scale (Frobenius norm)
    src_scale = torch.norm(src_centered, p='fro')  # scalar
    tgt_scale = torch.norm(tgt_centered, p='fro')  # scalar
    s = tgt_scale / src_scale  # scale factor

    # Compute rotation (SVD)
    H = src_centered.T @ tgt_centered  # [3, 3] covariance matrix
    U, S, V = torch.svd(H)  # SVD
    R = V @ U.T  # [3, 3] rotation matrix

    # Handle reflection case
    if torch.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T

    # Compute translation
    t = tgt_mean.squeeze(0) - s * (R @ src_mean.T).T.squeeze(0)  # [3]

    # add transform matrix to dict
    transform_matrix = torch.eye(4)
    transform_matrix[:3, :3] = s * R
    transform_matrix[:3, 3] = t

    return {
        "R": R,
        "t": t,
        "s": s,
        "transform_matrix": transform_matrix
    }


@torch.cuda.amp.autocast(enabled=False)
def compute_similarity_transform_from_bbox(canonical_bbox, transformed_pts, num_samples_per_dim=8):
    """
    Compute the similarity transform (R, t, s) from canonical bbox to transformed points.
    
    This function samples points from the canonical bbox, then uses those sampled points
    along with the corresponding transformed points to compute the similarity transform.
    
    Args:
        canonical_bbox: [2, 3] or [6] array/tensor, bounding box in canonical space
        transformed_pts: [num_samples_per_dim^3, 3] tensor, transformed points in target space
        num_samples_per_dim: number of samples along each dimension (default 8 for 8x8x8=512 points)
    
    Returns:
        dict with keys:
            R: [3, 3] rotation matrix
            t: [3] translation vector
            s: scalar scale factor
            transform_matrix: [4, 4] transformation matrix
    """
    # Sample points from canonical bbox
    src_pts = sample_points_from_bbox(canonical_bbox, num_samples_per_dim=num_samples_per_dim)
    
    # Ensure both are tensors with the same device/dtype
    if isinstance(transformed_pts, np.ndarray):
        transformed_pts = torch.from_numpy(transformed_pts).float()
    
    src_pts = src_pts.to(transformed_pts.device).to(transformed_pts.dtype)
    
    # Compute similarity transform
    return compute_similarity_transform(src_pts, transformed_pts)


def downsample_voxels(indices, voxel_res=64, factor=4, min_bound=-1):
    """
    indices: (N,3) 原 voxel indices，分辨率=voxel_res
    factor: 下采样倍率，例如4
    """
    # 原 voxel_size
    orig_voxel_size = 2.0 / voxel_res

    # 新 voxel grid 的分辨率与体素尺寸
    new_res = voxel_res // factor
    new_voxel_size = orig_voxel_size * factor

    # 计算新的 voxel index
    new_indices = indices // factor  # floor

    # voxel center
    # center = origin + (idx + 0.5) * new_voxel_size
    centers = min_bound + (new_indices + 0.5) * new_voxel_size

    return new_indices, centers


def transform_pcd_simple(transform_dict, pcd):
    # pcd: [N, 3], np.array
    if isinstance(transform_dict["R"], torch.Tensor):
        transform_dict = {k:v.cpu().numpy() for k,v in transform_dict.items()}
    transformed_pcd = (transform_dict["s"] * pcd @ transform_dict["R"].T) + transform_dict["t"]  # (N, 3)
    return transformed_pcd