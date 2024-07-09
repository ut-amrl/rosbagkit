import numpy as np
import cv2


def project_to_image(points, extrinsic, K, D, img_size):
    """
    Project the point cloud to the image plane

    Args:
        points: (N,3) The point cloud to project.
        extrinsic: (4,4) The camera extrinsic matrix.
        K: (3,3) The camera intrinsic matrix.
        D: (>=4,) The camera distortion coefficients.
        img_size: (2,) The size of the image. (height, width)

    Returns:
        projected_points: (M,2) The projected point cloud in the image plane.
        pc_depth: (M,) The depth values of the projected points.
        valid_indices: (M,) The indices of the valid points.
    """
    assert len(points) > 0 and points.shape[1] == 3, f"{points.shape} != (N,3)"
    assert extrinsic.shape == (4, 4), f"{extrinsic.shape} != (4,4)"
    assert K.shape == (3, 3), f"{K.shape} != (3,3)"
    assert D is None or len(D) >= 4, f"{len(D)} < 4"
    assert len(img_size) == 2, f"{len(img_size)} != 2"

    # Transform points to camera coordinate system (LiDAR -> Camera)
    points_cam = points @ extrinsic[:3, :3].T + extrinsic[:3, 3].T
    front_mask = points_cam[:, 2] > 0
    # front_mask = np.logical_and(points_cam[:, 2] > 0, points_cam[:, 2] < 10)

    # Project points onto the image plane
    projected_points, _ = cv2.projectPoints(
        points_cam[front_mask], np.zeros((3, 1)), np.zeros((3, 1)), K, D
    )
    projected_points = projected_points.reshape(-1, 2)

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


def project_to_rectified(points, extrinsic, P, img_size):
    """
    Project the point cloud to the rectified image

    Args:
        points: (N,3) The point cloud to project.
        extrinsic: (4,4) The camera extrinsic matrix.
        P: (3,4) The projection matrix.
        img_size: (2,) The size of the image. (imgH, imgW)

    Returns:
        projected_points: (M,2) The projected point cloud in the image plane.
        pc_depth: (M,) The depth values of the projected points.
        valid_indices: (M,) The indices of the valid points.
    """
    assert len(points) > 0 and points.shape[1] == 3, f"{points.shape} != (N,3)"
    assert extrinsic.shape == (4, 4), f"{extrinsic.shape} != (4,4)"
    assert P.shape == (3, 4), f"{P.shape} != (3,4)"
    assert len(img_size) == 2, f"{len(img_size)} != 2"

    # Transform points to camera coordinate system
    points_cam = points @ extrinsic[:3, :3].T + extrinsic[:3, 3].T
    front_mask = points_cam[:, 2] > 0
    # front_mask = np.logical_and(points_cam[:, 2] > 0, points_cam[:, 2] < 10)

    # Project points onto the rectified image plane (Proj * X)
    projected_points = points_cam[front_mask]
    projected_points = projected_points @ P[:, :3].T + P[:, 3].T
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
