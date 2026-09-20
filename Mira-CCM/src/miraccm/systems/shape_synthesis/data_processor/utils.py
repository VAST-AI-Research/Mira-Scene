from einops import rearrange
from UniDataset.utils.pcd_utils import compute_similarity_transform
import torch


def transform_pcd(latent_voxel_centers, latent_voxel_cam_pts, pcd):
    # latent_voxel_centers: [Np', 3], torch.float32
    # latent_voxel_cam_pts: [Np', 3], torch.float32
    # pcd: [N, 3], np.array
    transform = compute_similarity_transform(
        src=latent_voxel_centers,
        tgt=latent_voxel_cam_pts,
    )
    transform = {k:v.cpu().numpy() for k,v in transform.items()}
    transformed_pcd = (transform["s"] * pcd @ transform["R"].T) + transform["t"]  # (N, 3)
    return transformed_pcd


def transform_pcd_simple(transform_dict, pcd):
    # pcd: [N, 3], np.array
    if isinstance(transform_dict["R"], torch.Tensor):
        transform_dict = {k:v.cpu().numpy() for k,v in transform_dict.items()}
    transformed_pcd = (transform_dict["s"] * pcd @ transform_dict["R"].T) + transform_dict["t"]  # (N, 3)
    return transformed_pcd


def downsample_cube_pcd(cube_pcd, factor=2):
    # cube_pcd: [..., Np, 3]
    Np = cube_pcd.shape[-2]
    side_len = int(round(Np ** (1/3)))
    assert side_len ** 3 == Np, "Input cube_pcd is not cubic."
    new_side_len = side_len // factor
    new_Np = new_side_len ** 3
    cube_pcd = rearrange(cube_pcd, "... (D1 D2 D3) c -> ... D1 D2 D3 c", D1=side_len, D2=side_len, D3=side_len)
    downsampled_cube_pcd = cube_pcd[..., ::factor, ::factor, ::factor, :]
    downsampled_cube_pcd = rearrange(downsampled_cube_pcd, "... D1 D2 D3 c -> ... (D1 D2 D3) c")
    return downsampled_cube_pcd