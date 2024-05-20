import numpy as np

from utils.visualization import draw_points_on_image


def project_to_image(img, pc, cam_ext, K, D):
    """
    Project the point cloud to the image plane

    Args:
        img: The image to project the point cloud onto.
        pc: (N,3) The point cloud to project.
        cam_ext: (4,4) The camera extrinsic matrix.
        K: (3,3) The camera intrinsic matrix.
        D: (5,) The distortion coefficients.

    Returns:
        pc_img: (M,2) The projected point cloud in the image plane.
        pc_depth: (M,) The depth values of the projected points.
        valid_indices: (M,) The indices of the valid points.
    """
    H, W = img.shape[:2]

    # Transform points to camera coordinate system
    pc_cam = np.hstack((pc, np.ones((len(pc), 1))))
    pc_cam = pc_cam @ cam_ext[:3].T
    # 0 < z < 10
    valid_pc = np.logical_and(pc_cam[:, 2] > 0, pc_cam[:, 2] < 10)

    # Project points onto the image plane
    pc_img = pc_cam[valid_pc, :2] / pc_cam[valid_pc, 2, None]

    # Apply radial distortion
    x, y = pc_img[:, 0], pc_img[:, 1]
    r2 = x**2 + y**2
    radial = 1 + D[0] * r2 + D[1] * r2**2 + D[4] * r2**3
    tan_x = 2 * D[2] * x * y + D[3] * (r2 + 2 * x**2)
    tan_y = D[2] * (r2 + 2 * y**2) + 2 * D[3] * x * y
    x = x * radial + tan_x
    y = y * radial + tan_y
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


def project_to_rectified(img, pc, cam_ext, R, P, visualize=False):
    """
    Project the point cloud to the rectified image

    Args:
        img: The image to project the point cloud onto.
        pc: (N,3) The point cloud to project.
        cam_ext: (4,4) The camera extrinsic matrix.
        R: (3,3) The rectification matrix.
        P: (3,4) The projection matrix.

    Returns:
        pc_img: (M,2) The projected point cloud in the image plane.
        pc_depth: (M,) The depth values of the projected points.
        valid_indices: (M,) The indices of the valid points.
    """
    H, W = img.shape[:2]

    # Transform points to camera coordinate system
    pc_cam = np.hstack((pc, np.ones((len(pc), 1))))  # (N, 4) homogeneous coordinates
    pc_cam = pc_cam @ cam_ext[:3].T  # (N, 3) camera coordinates
    valid_pc = pc_cam[:, 2] > 0

    # Project points onto the image plane
    pc_img = pc_cam[valid_pc] @ R.T
    pc_img = np.hstack((pc_img, np.ones((len(pc_img), 1))))
    pc_img = pc_img @ P.T
    pc_img = pc_img[:, :2] / pc_img[:, 2, None]

    # Filter points outside the image
    in_bound = np.logical_and(
        np.logical_and(pc_img[:, 0] >= 0, pc_img[:, 0] < W),
        np.logical_and(pc_img[:, 1] >= 0, pc_img[:, 1] < H),
    )

    valid_indices = np.where(valid_pc)[0][in_bound]

    if visualize:
        draw_points_on_image(img, pc_img[in_bound], pc_cam[valid_indices, 2])

    return pc_img[in_bound], pc_cam[valid_indices, 2], valid_indices
