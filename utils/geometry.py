"""
Author:      Dongmyeong Lee (domlee[at]utexas.edu)
Date:        Sep 16, 2023
Description: functions for geometric operations.
"""

import os
import sys

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__)))
from utils.image import compute_iou


def get_corners_bbox_3d(bbox_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Get corners of the 3D bounding box in the local frame. The local frame is
    defined as the frame with the origin at the center of the bounding box and
    the x-axis pointing forward, y-axis to the left, and z-axis up.

    The 3D bounding box is defined as follows:

            w
       0 -------- 2
    l /|         /|                      z up   x front
     / |        / |                       ^      ^
    4 -------- 6  | h                     |     /
    |  |       |  |                       |    /
    |  |       |  |                       |   /
    |  1 ------|--3                       |  /
    | /        | /                        | /
    |/         |/                         |/
    5 -------- 7       y left <-----------O

    Args:
        bbox_3d: (9, ) array of [cX, cY, cZ, l, w, h, r, p, y]

    Returns:
        corners: (8, 3) array of 3D points
        edges: (12, 2) array of edges (corner1_idx, corner2_idx)
    """
    assert bbox_3d.shape == (9,), "Invalid shape of bbox_3d"
    cX, cY, cZ, l, w, h, r, p, y = bbox_3d

    # Create transformation matrix from the given r, p, y
    rot_mat = R.from_euler("xyz", [r, p, y], degrees=False).as_matrix()

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
        bbox_3d: (9, ) array of [cX, cY, cZ, l, w, h, r, p, y]
        transformation: (4, 4) transformation matrix

    Returns:
        transformed_bbox_3d: (9, ) array of [cX, cY, cZ, l, w, h, r, p, y]
    """
    assert bbox_3d.shape == (9,), "Invalid shape of bbox_3d"
    assert transformation.shape == (4, 4), "Invalid shape of transformation"
    cX, cY, cZ, l, w, h, r, p, y = bbox_3d

    bbox_frame = np.eye(4)
    bbox_frame[:3, 3] = np.array([cX, cY, cZ])
    bbox_frame[:3, :3] = R.from_euler("xyz", [r, p, y], degrees=False).as_matrix()

    transformed_bbox_frame = transformation @ bbox_frame

    transformed_bbox_3d = np.zeros(9)
    transformed_bbox_3d[:3] = transformed_bbox_frame[:3, 3]
    transformed_bbox_3d[3:6] = np.array([l, w, h])
    transformed_bbox_3d[6:9] = R.from_matrix(transformed_bbox_frame[:3, :3]).as_euler(
        "xyz", degrees=False
    )
    return transformed_bbox_3d


def project_points_3d_to_2d(
    points: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_size: Optional[np.ndarray] = None,
    dist_coeff: Optional[np.ndarray] = None,
    keep_size: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
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
        points_image: (M, 2) array of 2D points
        depths: (M, ) array of depths
        where M <= N is the number of points that are projected onto the image
        M == N if keep_size (NaN is used to replace non-projectable points)
    """
    assert points.ndim == 2 and points.shape[1] >= 3, "Invalid shape of points"

    if points.shape[0] < 1:
        return np.empty((0, 2)), np.empty((0,))

    # Transform points to the camera coordinate system
    points_homo = np.column_stack((points[:, :3], np.ones(points.shape[0])))
    points_camera = points_homo @ extrinsic[:3, :].T
    depths = points_camera[:, 2]

    # Filter out points with negative depth
    mask_visible = depths > 0
    points_camera = points_camera[mask_visible]
    depths = depths[mask_visible]

    # Project Points
    points_image = points_camera @ intrinsic.T
    points_image = points_image / points_image[:, 2].reshape(-1, 1)

    # TODO: handle distortion (if dist_coeff is provided)

    # Filter points within the image boundaries
    if image_size is not None:
        mask_in_image = (
            (points_image[:, 0] >= 0)
            & (points_image[:, 0] < image_size[0])
            & (points_image[:, 1] >= 0)
            & (points_image[:, 1] < image_size[1])
        )
        mask_visible[mask_visible] = mask_in_image
        points_image = points_image[mask_in_image]
        depths = depths[mask_in_image]

    if keep_size:
        resized_points = np.full((points.shape[0], 2), np.nan)
        resized_depths = np.full(points.shape[0], np.nan)
        resized_points[mask_visible] = points_image
        resized_depths[mask_visible] = depths
        return resized_points, resized_depths

    return points_image, depths


def filter_points_inside_bbox_3d(points: np.ndarray, bbox_3d: np.ndarray) -> np.ndarray:
    """
    Filter out points that are inside the 3D bounding box.

    Args:
        points: (N, 3+) array of 3D points
        bbox_3d: (9,) array of [cX, cY, cZ, l, w, h, r, p, y]

    Returns:
        points: (M, 3) array of 3D points
        where M <= N is the number of points that are inside the bounding box
    """
    assert bbox_3d.shape == (9,), "Invalid shape of bbox_3d"
    cX, cY, cZ, l, w, h, r, p, y = bbox_3d

    # Transform points to the bounding box coordinate system
    rot_mat = R.from_euler("xyz", [r, p, y], degrees=False).as_matrix()
    transformed_points = (points[:, :3] - np.array([cX, cY, cZ])) @ rot_mat

    # Filter out points that are inside the bounding box
    hh, hl, hw = h / 2.0, l / 2.0, w / 2.0
    mask_inside = (
        (transformed_points[:, 0] >= -hl)
        & (transformed_points[:, 0] <= hl)
        & (transformed_points[:, 1] >= -hw)
        & (transformed_points[:, 1] <= hw)
        & (transformed_points[:, 2] >= -hh)
        & (transformed_points[:, 2] <= hh)
    )
    return points[mask_inside]
