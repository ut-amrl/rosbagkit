import sys
import pathlib
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from utils.visualization import visualize_pointcloud


def project_to_image(img, pc, cam_ext, K, D):
    H, W = img.shape[:2]

    # Transform points to camera coordinate system
    pc_world = np.hstack((pc, np.ones((pc.shape[0], 1))))
    pc_cam = pc_world @ cam_ext[:3].T  # P_cam = H_wc * P_world
    valid_pc = pc_cam[:, 2] > 0

    # Project points onto the image plane
    pc_img = pc_cam[valid_pc, :2] / pc_cam[valid_pc, 2, None]

    # Apply radial distortion
    x, y = pc_img[:, 0], pc_img[:, 1]
    # TODO: fix the radial distortion
    # r2 = x**2 + y**2
    # radial = 1 + D[0] * r2 + D[1] * r2**2 + D[4] * r2**3
    # tan_x = 2 * D[2] * x * y + D[3] * (r2 + 2 * x**2)
    # tan_y = D[2] * (r2 + 2 * y**2) + 2 * D[3] * x * y
    # x = x * radial + tan_x
    # y = y * radial + tan_y
    pc_img = np.vstack((x, y, np.ones_like(x))).T

    # Apply camera matrix
    pc_img = pc_img @ K[:2].T

    # Filter points outside the image
    in_bound = np.logical_and(
        np.logical_and(pc_img[:, 0] >= 0, pc_img[:, 0] < W),
        np.logical_and(pc_img[:, 1] >= 0, pc_img[:, 1] < H),
    )

    valid_indices = np.where(valid_pc)[0][in_bound]

    return pc_img[in_bound, :2], pc_cam[valid_indices, 2], valid_indices
