from typing import Optional
import numpy as np


def project_points(
    points_3d: np.ndarray,
    extrinsic: np.ndarray,
    projection: np.ndarray,
    img_size: Optional[tuple] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project the point cloud to the rectified image

    Args:
        points_3d: (N,3) The point cloud to project.
        extrinsic: (4,4) The camera extrinsic matrix.
        projection: (3,4) The projection matrix.
        img_size: (optional) (2,) The size of the image. (imgH, imgW)

    Returns:
        projected_points: (M,2) The projected point cloud in the image plane.
        pc_depth: (M,) The depth values of the projected points.
        valid_indices: (M,) The indices of the valid points.
    """
    assert len(points_3d) > 0 and points_3d.shape[1] == 3, f"{points_3d.shape} != (N,3)"
    assert extrinsic.shape == (4, 4), f"{extrinsic.shape} != (4,4)"
    assert projection.shape == (3, 4), f"{projection.shape} != (3,4)"
    assert img_size is None or len(img_size) == 2, f"{img_size} != (2,)"

    if len(points_3d) == 0:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)

    # Transform points to camera coordinate system
    points_cam = points_3d @ extrinsic[:3, :3].T + extrinsic[:3, 3].T
    front_mask = points_cam[:, 2] > 0

    # Project points onto the image plane (Proj * X)
    projected_points = points_cam[front_mask]
    projected_points = projected_points @ projection[:, :3].T + projection[:, 3].T
    projected_points = projected_points[:, :2] / projected_points[:, 2, None]

    if img_size is None:
        valid_indices = np.where(front_mask)[0]
        return projected_points, points_cam[valid_indices, 2], valid_indices

    # Filter points outside the image
    in_bound = np.logical_and(
        np.logical_and(
            projected_points[:, 0] >= 0, projected_points[:, 0] < img_size[1]
        ),
        np.logical_and(
            projected_points[:, 1] >= 0, projected_points[:, 1] < img_size[0]
        ),
    )
    valid_indices = np.where(front_mask)[0][in_bound]
    return projected_points[in_bound], points_cam[valid_indices, 2], valid_indices
