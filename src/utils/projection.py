import numpy as np


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
    valid_pc = pc_cam[:, 2] > 0

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


def project_to_rectified(points, extrinsic, rect_mat, proj_mat, img_size):
    """
    Project the point cloud to the rectified image

    Args:
        points: (N,3) The point cloud to project.
        extrinsic: (4,4) The camera extrinsic matrix.
        rect_mat: (3,3) The rectification matrix.
        proj_mat: (3,4) The projection matrix.
        img_size: (2,) The size of the image. (H, W)

    Returns:
        pc_img: (M,2) The projected point cloud in the image plane.
        pc_depth: (M,) The depth values of the projected points.
        valid_indices: (M,) The indices of the valid points.
    """
    assert len(points) > 0 and points.shape[1] == 3, f"{points.shape} != (N,3)"
    assert extrinsic.shape == (4, 4), f"{extrinsic.shape} != (4,4)"
    assert rect_mat.shape == (3, 3), f"{rect_mat.shape} != (3,3)"
    assert proj_mat.shape == (3, 4), f"{proj_mat.shape} != (3,4)"
    assert len(img_size) == 2, f"{len(img_size)} != 2"

    imgH, imgW = img_size

    # Transform points to camera coordinate system
    points_cam = points @ extrinsic[:3, :3].T + extrinsic[:3, 3].T
    front_mask = points_cam[:, 2] > 0

    # Project points onto the rectified image plane (Proj * Rect * X)
    projected_points = points_cam[front_mask] @ rect_mat.T
    projected_points = projected_points @ proj_mat[:, :3].T + proj_mat[:, 3].T
    projected_points = projected_points[:, :2] / projected_points[:, 2, None]

    # Filter points outside the image
    in_bound = np.logical_and(
        np.logical_and(projected_points[:, 0] >= 0, projected_points[:, 0] < imgW),
        np.logical_and(projected_points[:, 1] >= 0, projected_points[:, 1] < imgH),
    )

    valid_indices = np.where(front_mask)[0][in_bound]

    return projected_points[in_bound], points_cam[valid_indices, 2], valid_indices
