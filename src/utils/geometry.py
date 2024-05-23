"""
Author:      Dongmyeong Lee (domlee[at]utexas.edu)
Date:        Sep 16, 2023
Description: functions for geometric operations.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional


def get_3d_bbox_corners(
    bbox: np.ndarray, degrees: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get corners of the 3D bounding box in the global frame. The local frame is
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
    assert bbox.shape == (9,), f"{bbox.shape} != (9,)"
    cX, cY, cZ, l, w, h, r, p, y = bbox

    # Create transformation matrix from the given r, p, y
    rot_mat = R.from_euler("xyz", [r, p, y], degrees=degrees).as_matrix()

    # Half-length, half-width, and half-height
    hh, hl, hw = h / 2.0, l / 2.0, w / 2.0

    # Define 8 corners of the bounding box in the local frame
    # fmt: off
    local_corners = np.array([
        [hl, hw, hh], [hl, hw, -hh], [hl, -hw, hh], [hl, -hw, -hh],
        [-hl, hw, hh], [-hl, hw, -hh], [-hl, -hw, hh], [-hl, -hw, -hh]
    ])  # fmt: on

    # Transform corners to the global frame
    frame_corners = local_corners @ rot_mat.T + np.array([cX, cY, cZ])

    # fmt: off
    edges = np.array([(0, 1), (1, 3), (3, 2), (2, 0),
                      (4, 5), (5, 7), (7, 6), (6, 4),
                      (0, 4), (1, 5), (2, 6), (3, 7)])  # fmt: on
    return frame_corners, edges


def get_3d_bbox_planes(bbox: np.ndarray, degrees: bool = False) -> np.ndarray:
    """
    Get the 3D bounding box planes in the global frame. The local frame is
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
        planes: (6, 4) array of 3D planes
    """
    assert bbox.shape == (9,), f"{bbox.shape} != (9,)"
    corners, _ = get_3d_bbox_corners(bbox, degrees=degrees)

    planes_coners = np.array(
        [
            (0, 1, 3, 2),  # front
            (4, 5, 7, 6),  # back
            (0, 1, 5, 4),  # left
            (2, 3, 7, 6),  # right
            (0, 2, 6, 4),  # top
            (1, 3, 7, 5),  # bottom
        ]
    )

    # Define 6 planes of the bounding box
    planes = np.zeros((6, 4))
    for i in range(6):
        p1, p2, p3, p4 = corners[planes_coners[i]]
        normal = np.cross(p3 - p1, p4 - p2)
        normal /= np.linalg.norm(normal)
        planes[i, :3] = normal
        planes[i, 3] = -np.dot(normal, p1)

    return planes


def transform_3d_bbox(
    bbox: np.ndarray, transformation: np.ndarray, degrees: bool = False
) -> np.ndarray:
    """
    Transform a 3D bounding box with the given transformation matrix.
    example: bbox_3d in LiDAR frame -> bbox_3d in map frame,
             transformation should be LiDAR to map transformation matrix

    Args:
        bbox: (9, ) array of [cX, cY, cZ, l, w, h, r, p, y]
        transformation: (4, 4) transformation matrix
        degrees: bool, if True, r, p, y are in degrees

    Returns:
        transformed_bbox_3d: (9, ) array of [cX, cY, cZ, l, w, h, r, p, y]
    """
    assert bbox.shape == (9,), f"{bbox.shape} != (9, )"
    assert transformation.shape == (4, 4), f"{transformation.shape} != (4, 4)"
    cX, cY, cZ, l, w, h, r, p, y = bbox

    # Tlb
    bbox_frame = np.eye(4)
    bbox_frame[:3, 3] = np.array([cX, cY, cZ])
    bbox_frame[:3, :3] = R.from_euler("xyz", [r, p, y], degrees=degrees).as_matrix()

    # Twb = Twl @ Tlb
    transformed_bbox_frame = transformation @ bbox_frame

    transformed_bbox_3d = np.zeros(9)
    transformed_bbox_3d[:3] = transformed_bbox_frame[:3, 3]
    transformed_bbox_3d[3:6] = np.array([l, w, h])
    transformed_bbox_3d[6:9] = R.from_matrix(transformed_bbox_frame[:3, :3]).as_euler(
        "xyz", degrees=degrees
    )
    return transformed_bbox_3d


def transform_plane(plane: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    """
    Transform a 3D plane with the given transformation matrix.
    https://stackoverflow.com/questions/7685495/transforming-a-3d-plane-using-a-4x4-matrix

    Args:
        plane: (4, ) array of [a, b, c, d]; plane equation (ax + by + cz + d = 0)
        transformation: (4, 4) transformation matrix

    Returns:
        transformed_plane: (4, ) array of [a, b, c, d] transformed plane equation
    """
    assert plane.shape == (4,), f"{plane.shape} != (4,)"
    assert transformation.shape == (4, 4), f"{transformation.shape} != (4, 4)"

    transformed_plane = plane @ np.linalg.inv(transformation)
    transformed_plane /= np.linalg.norm(transformed_plane[:3])

    return transformed_plane


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


if __name__ == "__main__":
    plane = np.array([1, 0, 0, -1])
    transformation = np.eye(4)
    transformation[:3, 3] = np.array([1, 2, 3])

    transformed_plane = transform_plane(plane, np.linalg.inv(transformation))
    print(transformed_plane)
