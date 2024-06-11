import numpy as np


def project_to_image(points, extrinsic, intrinsic, img_size, dist_coeffs=None):
    """
    Project the point cloud to the image plane

    Args:
        points: (N,3) The point cloud to project.
        extrinsic: (4,4) The camera extrinsic matrix.
        intrinsic: (3,3) The camera intrinsic matrix.
        img_size: (2,) The size of the image. (height, width)
        dist_coeffs: (5,) The camera distortion coefficients.

    Returns:
        projected_points: (M,2) The projected point cloud in the image plane.
        pc_depth: (M,) The depth values of the projected points.
        valid_indices: (M,) The indices of the valid points.
    """
    assert len(points) > 0 and points.shape[1] == 3, f"{points.shape} != (N,3)"
    assert extrinsic.shape == (4, 4), f"{extrinsic.shape} != (4,4)"
    assert intrinsic.shape == (3, 3), f"{intrinsic.shape} != (3,3)"
    assert dist_coeffs is None or len(dist_coeffs) == 5, f"{len(dist_coeffs)} != 5"
    assert len(img_size) == 2, f"{len(img_size)} != 2"

    # Transform points to camera coordinate system (LiDAR -> Camera)
    points_cam = points @ extrinsic[:3, :3].T + extrinsic[:3, 3].T
    # front_mask = points_cam[:, 2] > 0
    front_mask = np.logical_and(points_cam[:, 2] > 0, points_cam[:, 2] < 10)

    # Project points onto the image plane
    projected_points = points_cam[front_mask, :] / points_cam[front_mask, 2, None]

    # Apply radial distortion
    if dist_coeffs is not None:
        x, y = projected_points[:, 0], projected_points[:, 1]
        r2 = x**2 + y**2
        radial = (
            1 + dist_coeffs[0] * r2 + dist_coeffs[1] * r2**2 + dist_coeffs[4] * r2**3
        )
        tan_x = 2 * dist_coeffs[2] * x * y + dist_coeffs[3] * (r2 + 2 * x**2)
        tan_y = dist_coeffs[2] * (r2 + 2 * y**2) + 2 * dist_coeffs[3] * x * y
        x = x * radial + tan_x
        y = y * radial + tan_y
        projected_points = np.vstack((x, y, np.ones_like(x))).T

    # Apply camera matrix
    projected_points = projected_points @ intrinsic[:2].T

    # Filter points outside the image
    in_bound = np.logical_and(
        np.logical_and(
            projected_points[:, 0] >= 0, projected_points[:, 0] < img_size[1]  # imgW
        ),
        np.logical_and(
            projected_points[:, 1] >= 0, projected_points[:, 1] < img_size[0]  # imgH
        ),
    )

    valid_indices = np.where(front_mask)[0][in_bound]

    return projected_points[in_bound, :2], points_cam[valid_indices, 2], valid_indices


def project_to_rectified(points, extrinsic, proj_mat, img_size):
    """
    Project the point cloud to the rectified image

    Args:
        points: (N,3) The point cloud to project.
        extrinsic: (4,4) The camera extrinsic matrix.
        proj_mat: (3,4) The projection matrix.
        img_size: (2,) The size of the image. (imgH, imgW)

    Returns:
        projected_points: (M,2) The projected point cloud in the image plane.
        pc_depth: (M,) The depth values of the projected points.
        valid_indices: (M,) The indices of the valid points.
    """
    assert len(points) > 0 and points.shape[1] == 3, f"{points.shape} != (N,3)"
    assert extrinsic.shape == (4, 4), f"{extrinsic.shape} != (4,4)"
    assert proj_mat.shape == (3, 4), f"{proj_mat.shape} != (3,4)"
    assert len(img_size) == 2, f"{len(img_size)} != 2"

    # Transform points to camera coordinate system
    points_cam = points @ extrinsic[:3, :3].T + extrinsic[:3, 3].T
    # front_mask = points_cam[:, 2] > 0
    front_mask = np.logical_and(points_cam[:, 2] > 0, points_cam[:, 2] < 10)

    # Project points onto the rectified image plane (Proj * Rect * X)
    projected_points = points_cam[front_mask]
    projected_points = projected_points @ proj_mat[:, :3].T + proj_mat[:, 3].T
    projected_points = projected_points[:, :2] / projected_points[:, 2, None]

    # Filter points outside the image
    in_bound = np.logical_and(
        np.logical_and(
            projected_points[:, 0] >= 0, projected_points[:, 0] < img_size[1]  # imgW
        ),
        np.logical_and(
            projected_points[:, 1] >= 0, projected_points[:, 1] < img_size[0]  # imgH
        ),
    )

    valid_indices = np.where(front_mask)[0][in_bound]

    return projected_points[in_bound], points_cam[valid_indices, 2], valid_indices
