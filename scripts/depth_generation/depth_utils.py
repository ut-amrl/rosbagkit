from math import floor, ceil

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from camera.projection import project_points


def project_volume_to_depth(
    img: np.ndarray,
    accumulated_pc: list[np.ndarray],
    extrinsic: np.ndarray,
    projection: np.ndarray,
    volume: float,
    filter_size: int = 0,
    visualize: bool = False,
):
    """
    Project pointclouds to the image plane and compute the depth image,
    assuming points have volume.
    """
    assert extrinsic.shape == (4, 4), f"{extrinsic.shape} != (4,4)"
    assert projection.shape == (3, 4), f"{projection.shape} != (3,4)"
    imgH, imgW = img.shape[:2]
    depth = np.zeros((imgH, imgW), dtype=np.float32)

    all_pc_img = []
    all_pc_depth = []

    # project the pointclouds to the image plane (assume points with volume)
    for pc in accumulated_pc:
        # project the pointcloud to the rectified image plane
        pc_img, pc_depth, _ = project_points(pc, extrinsic, projection, (imgH, imgW))
        all_pc_img.append(pc_img)
        all_pc_depth.append(pc_depth)

        # sort the points by depth (from near to far)
        sort_idx = np.argsort(pc_depth)

        # compute the voxel size in the image plane (occlusion-aware depth estimation)
        dx = projection[0, 0] * volume / pc_depth
        dy = projection[1, 1] * volume / pc_depth

        for i in sort_idx:
            min_x = max(0, floor(pc_img[i, 0] - dx[i] / 2))
            min_y = max(0, floor(pc_img[i, 1] - dy[i] / 2))
            max_x = min(imgW, ceil(pc_img[i, 0] + dx[i] / 2))
            max_y = min(imgH, ceil(pc_img[i, 1] + dy[i] / 2))

            # Use broadcasting to efficiently update the depth map
            mask = depth[min_y:max_y, min_x:max_x] == 0
            depth[min_y:max_y, min_x:max_x][mask] = pc_depth[i]

    # # fill the holes in the depth image
    # depth_filled = depth.copy()
    # if filter_size > 0:
    #     hole_mask = depth == 0
    #     max_values = maximum_filter(depth, size=filter_size)
    #     depth_filled[hole_mask] = max_values[hole_mask]
    #     print(f"number of holes: {np.sum(hole_mask)}")
    #     print(f"number of filled holes: {np.sum(hole_mask & (depth_filled != 0))}")

    if visualize:
        fig, ax = plt.subplots(1, 3, figsize=(15, 7))

        ax[0].imshow(img)

        ax[1].imshow(img)
        for pc_img, pc_depth in zip(all_pc_img, all_pc_depth):
            ax[1].scatter(pc_img[:, 0], pc_img[:, 1], s=1, c=pc_depth, cmap="jet")
        ax[1].set_xlim(0, imgW)
        ax[1].set_ylim(imgH, 0)
        ax[1].set_title("Projected Point Clouds")

        jet_cmap = plt.cm.get_cmap("jet", 256)
        newcolors = jet_cmap(np.linspace(0, 1, 256))
        newcolors[0, :] = np.array(
            [1, 1, 1, 1]
        )  # Set the color for zero values to white
        custom_cmap = ListedColormap(newcolors)
        masked_depth = np.ma.masked_where(depth == 0, depth)
        # masked_depth_filled = np.ma.masked_where(depth_filled == 0, depth_filled)

        ax[2].imshow(masked_depth, cmap="jet")
        ax[2].set_title("Depth Image")

        # ax[3].imshow(masked_depth_filled, cmap="jet")
        # ax[3].set_title("Filled Depth Image")

        plt.show()

    return depth


def fill_depth_bins(depth_bins, pc_img, pc_depths, option="max"):
    """
    Fill the depth bins with the maximum depth value for each pixel.

    Args:
        depth_bins: depth array to fill (H, W, 3) -> (x, y, depth) (float)
        pc_img: The projected point cloud image coordinates. (x, y) float values.
        pc_depths: The point cloud depth values.
    """
    assert depth_bins.ndim == 3 and depth_bins.shape[2] == 3
    assert len(pc_img) == len(pc_depths)

    H, W = depth_bins.shape[:2]

    # Convert the projected point cloud to integer indices
    pc_img_int = np.round(pc_img).astype(int)

    # Filter points outside the image
    mask = np.logical_and(
        np.logical_and(pc_img_int[:, 0] >= 0, pc_img_int[:, 0] < W),
        np.logical_and(pc_img_int[:, 1] >= 0, pc_img_int[:, 1] < H),
    )
    pc_img = pc_img[mask]
    pc_img_int = pc_img_int[mask]
    pc_depths = pc_depths[mask]

    # Fill the depth bins with the maximum depth value
    new_depth_bins = np.full((H, W, 3), np.nan, dtype=np.float32)
    for idx, ((x, y), depth) in enumerate(zip(pc_img_int, pc_depths)):
        if (
            np.isnan(depth_bins[y, x, 2])
            or (option == "max" and depth > depth_bins[y, x, 2])
            or (option == "min" and depth < depth_bins[y, x, 2])
        ):
            new_depth_bins[y, x] = np.array([pc_img[idx, 0], pc_img[idx, 1], depth])

    # Update the depth bins
    updated_depth_bins = depth_bins.copy()
    mask = ~(np.isnan(new_depth_bins[:, :, 2]) | np.isinf(new_depth_bins[:, :, 2]))
    updated_depth_bins[mask] = new_depth_bins[mask]

    return updated_depth_bins


def densify_depth(depth_bins, grid=5):
    """
    Densify the depth image using inverse distance weighted interpolation.

    Args:
        depth_bins: The depth bins array (H, W, 3) -> (x, y, depth)
        grid: The grid size for the interpolation.
    """
    assert depth_bins.ndim == 3 and depth_bins.shape[2] == 3

    H, W, _ = depth_bins.shape

    ng = 2 * grid + 1
    mX = np.full((H, W), np.inf)
    mY = np.full((H, W), np.inf)
    mD = np.zeros((H, W))

    # mask for valid depth values (nan, inf)
    valid_mask = ~(np.isnan(depth_bins[:, :, 2]) | np.isinf(depth_bins[:, :, 2]))
    valid_points = depth_bins[valid_mask]
    y_indices, x_indices = np.where(valid_mask)

    # Get the relative coordinates
    mX[y_indices, x_indices] = valid_points[:, 0] - x_indices
    mY[y_indices, x_indices] = valid_points[:, 1] - y_indices
    mD[y_indices, x_indices] = valid_points[:, 2]

    # Initialize the kernel matrices
    KmX = np.zeros((ng, ng, H - ng + 1, W - ng + 1))
    KmY = np.zeros((ng, ng, H - ng + 1, W - ng + 1))
    KmD = np.zeros((ng, ng, H - ng + 1, W - ng + 1))

    for i in range(ng):
        for j in range(ng):
            KmX[i, j] = mX[i : (H - ng + i + 1), j : (W - ng + j + 1)] - grid + i
            KmY[i, j] = mY[i : (H - ng + i + 1), j : (W - ng + j + 1)] - grid + j
            KmD[i, j] = mD[i : (H - ng + i + 1), j : (W - ng + j + 1)]

    # Calculate the Inverse Distance Weight
    S = np.zeros_like(KmD[0, 0])
    Y = np.zeros_like(KmD[0, 0])
    for i in range(ng):
        for j in range(ng):
            s = 1 / (np.sqrt(KmX[i, j] ** 2 + KmY[i, j] ** 2) + 1e-12)
            Y += s * KmD[i, j]
            S += s

    S[S == 0] = 1
    densified_depth = np.zeros((H, W))
    densified_depth[grid:-grid, grid:-grid] = Y / S
    return densified_depth


def alternating_indices(window_size):
    """Generate alternating indices from -window_size//2 to window_size//2

    Example: window_size=7 -> [0, -1, 1, -2, 2, -3, 3]
    """
    yield 0
    for i in range(1, window_size // 2 + 1):
        yield -i
        yield i

