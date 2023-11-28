
"""
Author:      Dongmyeong Lee (domlee[at]utexas.edu)
Date:        September 16, 2023
Description: functions for geometric operations.
"""
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple


def get_corners_bbox_3d(bbox_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get corners of the 3D bounding box in the local frame. The local frame is
    defined as the frame with the origin at the center of the bounding box and
    the x-axis pointing forward, y-axis to the left, and z-axis up.

    The 3D bounding box is defined as follows:

             l
       0 -------- 2
    w /|         /|                      z up   x front
     / |        / |                       ^      ^
    4 -------- 6  | h                     |     /
    |  |       |  |                       |    /
    |  |       |  |                       |   /
    |  1 ------|--3                       |  /
    | /        | /                        | /
    |/         |/                         |/
    5 -------- 7       y left <-----------O

    Args:
        bbox_3d: (9, ) array of [cX, cY, cZ, h, l, w, roll, pitch, yaw]

    Returns:
        corners: (8, 3) array of 3D points
        edges: (12, 2) array of edges (corner1_idx, corner2_idx)
    """
    assert bbox_3d.shape == (9,), "Invalid shape of bbox_3d"
    cX, cY, cZ, h, l, w, roll, pitch, yaw = bbox_3d

    # Create transformation matrix from the given roll, pitch, yaw
    rot_mat = R.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_matrix()

    # Half-length, half-width, and half-height
    hh, hl, hw = h / 2.0, l / 2.0, w / 2.0

    # Define 8 corners of the bounding box in the local frame
    # fmt: off
    local_corners = np.array([
        [hl, hw, hh], [hl, hw, -hh], [hl, -hw, hh], [hl, -hw, -hh],
        [-hl, hw, hh], [-hl, hw, -hh], [-hl, -hw, hh], [-hl, -hw, -hh]
    ])  # fmt: on

    # Transform corners to the frame
    frame_corners = local_corners @ rot_mat.T
    frame_corners += np.array([cX, cY, cZ])

    # fmt: off
    edges = np.array([(0, 1), (1, 3), (3, 2), (2, 0),
                      (4, 5), (5, 7), (7, 6), (6, 4),
                      (0, 4), (1, 5), (2, 6), (3, 7)])  # fmt: on
    return frame_corners, edges


def transform_bbox_3d(bbox_3d: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    """
    Transform a 3D bounding box with the given transformation matrix.
    example: bbox_3d in LiDAR frame -> bbox_3d in map frame,
             transformation should be LiDAR to map transformation matrix

    Args:
        bbox_3d: (9, ) array of [cX, cY, cZ, h, l, w, roll, pitch, yaw]
        transformation: (4, 4) transformation matrix

    Returns:
        transformed_bbox_3d: (9, ) array of [cX, cY, cZ, h, l, w, roll, pitch, yaw]
    """
    assert bbox_3d.shape == (9,), "Invalid shape of bbox_3d"
    assert transformation.shape == (4, 4), "Invalid shape of transformation"

    bbox_frame = np.eye(4)
    bbox_frame[:3, 3] = np.array(bbox_3d[:3])
    bbox_frame[:3, :3] = R.from_euler("xyz", bbox_3d[6:9], degrees=False).as_matrix()

    transformed_bbox_frame = transformation @ bbox_frame

    transformed_bbox_3d = np.zeros(9)
    transformed_bbox_3d[:3] = transformed_bbox_frame[:3, 3]
    transformed_bbox_3d[3:6] = bbox_3d[3:6]
    transformed_bbox_3d[6:9] = R.from_matrix(transformed_bbox_frame[:3, :3]).as_euler(
        "xyz", degrees=False
    )
    return transformed_bbox_3d


def project_bbox_3d_to_2d(
    bbox_3d: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_size: np.ndarray,
    dist_coeff: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Project 3D bounding box onto the image plane. If the bounding box is
    behind the camera, compute the intersection of the bounding box with the
    camera x-y plane and project the intersection points onto the image plane.

    Args:
        bbox_3d: (9,) array of [cX, cY, cZ, h, l, w, roll, pitch, yaw]
        extrinsic: (4, 4) extrinsic matrix
        intrinsic: (3, 3) intrinsic matrix
        image_size: (2,) image size (width, height) in pixels
        dist_coeff: (5,) dist_coeff coefficients (k1, k2, p1, p2, k3)

    Returns:
        x1, y1, x2, y2: (4,) coordinates of the bounding box (in pixels)
    """
    # Get corners of the bounding box in the LiDAR coordinate system
    corners, edges = get_corners_bbox_3d(bbox_3d)

    # Transform corners to the camera coordinate system
    corners_homo = np.column_stack((corners, np.ones(corners.shape[0])))
    corners_camera = corners_homo @ extrinsic[:3, :].T

    # Intersect the bounding box with the camera plane where z = 1e-3
    z_plane = 1e-3
    intersection_points = []
    for edge in edges:
        p1, p2 = corners_camera[edge]
        if (p1[2] - z_plane) * (p1[2] - z_plane) < 0:
            t = (p1[2] - z_plane) / (p2[2] - p1[2])
            intersection_point = (1 - t) * p1 + t * p2
            intersection_points.append(intersection_point)

    if intersection_points:
        points_camera = np.vstack((corners_camera, np.array(intersection_points)))
    else:
        points_camera = corners_camera

    # points_camera = np.vstack((corners_camera, intersection_points))
    points_image, _ = project_points_3d_to_2d(
        corners_camera, np.eye(4), intrinsic, image_size, dist_coeff, keep_size=False
    )

    if points_image.size == 0:
        return None

    x1, y1 = np.min(points_image, axis=0)
    x2, y2 = np.max(points_image, axis=0)
    return np.array([x1, y1, x2, y2])


def project_points_3d_to_2d(
    points: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_size: Optional[np.ndarray] = None,
    dist_coeff: Optional[np.ndarray] = None,
    keep_size: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points onto the image plane and filter out points that are
    outside the image boundaries.

    Args:
        points: (N, 3) array of 3D points
        extrinsic: (4, 4) extrinsic matrix
        intrinsic: (3, 3) intrinsic matrix
        image_size: (2, ) image size (width, height) in pixels
        or None to clip points later
        dist_coeff: (5, ) dist_coeff coefficients (k1, k2, p1, p2, k3)
        keep_size: whether to keep the size of points array

    Returns:
        image_points: (M, 2) array of 2D points
        depths: (M, ) array of depths
        where M <= N is the number of points that are projected onto the image
        M == N if keep_size (NaN is used to replace non-projectable points)
    """
    assert points.ndim == 2 and points.shape[1] == 3, "Invalid shape of points"

    if points.shape[0] < 1:
        return np.empty((0, 2)), np.empty((0,))

    # Transform points to the camera coordinate system
    points_homo = np.column_stack((points, np.ones(points.shape[0])))
    camera_points = points_homo @ extrinsic[:3, :].T
    depths = camera_points[:, 2]

    # Filter out points with negative depth
    mask_visible = depths > 0
    if np.any(mask_visible):
        # Project Points
        image_points, _ = cv2.projectPoints(
            camera_points[mask_visible], np.zeros(3), np.zeros(3), intrinsic, dist_coeff
        )
        image_points = image_points.squeeze().reshape(-1, 2)
    else:
        return np.empty((0, 2)), np.empty((0,))

    # Filter points within the image boundaries
    if image_size is not None:
        mask_in_image = (
            (image_points[:, 0] >= 0)
            & (image_points[:, 0] < image_size[0])
            & (image_points[:, 1] >= 0)
            & (image_points[:, 1] < image_size[1])
        )
        image_points = image_points[mask_in_image]
        depths = depths[mask_visible][mask_in_image]
        mask_visible[mask_visible] = mask_in_image

    if keep_size:
        resized_points = np.full((points.shape[0], 2), np.nan)
        resized_depths = np.full(points.shape[0], np.nan)
        resized_points[mask_visible] = image_points
        resized_depths[mask_visible] = depths
        return resized_points, resized_depths

    return image_points, depths


def filter_points_inside_bbox_3d(
    points: np.ndarray,
    bbox_3d: np.ndarray,
) -> np.ndarray:
    """
    Filter out points that are inside the 3D bounding box.

    Args:
        points: (N, 3+) array of 3D points
        bbox_3d: (9,) array of [cX, cY, cZ, h, l, w, roll, pitch, yaw]

    Returns:
        points: (M, 3) array of 3D points
        where M <= N is the number of points that are inside the bounding box
    """
    rot_mat = R.from_euler("xyz", bbox_3d[6:9], degrees=False).as_matrix()

    # Transform points to the bounding box coordinate system
    transformed_points = (points[:,:3] - bbox_3d[:3]) @ rot_mat

    # Filter out points that are inside the bounding box
    mask_inside = (
        (transformed_points[:, 0] >= -bbox_3d[4] / 2.0)
        & (transformed_points[:, 0] <= bbox_3d[4] / 2.0)
        & (transformed_points[:, 1] >= -bbox_3d[5] / 2.0)
        & (transformed_points[:, 1] <= bbox_3d[5] / 2.0)
        & (transformed_points[:, 2] >= -bbox_3d[3] / 2.0)
        & (transformed_points[:, 2] <= bbox_3d[3] / 2.0)
    )
    return points[mask_inside]

