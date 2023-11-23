"""
Author:      Dongmyeong Lee (domlee[at]utexas.edu)
Date:        September 16, 2023
Description: functions for geometric operations.
"""
from typing import Optional, Tuple

import numpy as np
import cv2

from scipy.spatial.transform import Rotation as R


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


# def project_points_3d_to_2d(
# points: np.ndarray,
# extrinsic: np.ndarray,
# intrinsic: np.ndarray,
# image_size: Optional[np.ndarray] = None,
# dist_coeff: Optional[np.ndarray] = None,
# keep_size: bool = False,
# ) -> Tuple[np.ndarray, np.ndarray]:
# """
# Project 3D points onto the image plane and filter out points that are
# outside the image boundaries.

# Args:
# points: (N, 3) array of 3D points
# extrinsic: (4, 4) extrinsic matrix
# intrinsic: (3, 3) intrinsic matrix
# image_size: (2, ) image size (width, height) in pixels
# or None to clip points later
# dist_coeff: (5, ) dist_coeff coefficients (k1, k2, p1, p2, k3)
# keep_size: whether to keep the size of points array

# Returns:
# image_points: (M, 2) array of 2D points
# depths: (M, ) array of depths
# where M <= N is the number of points that are projected onto the image
# M == N if keep_size (NaN is used to replace non-projectable points)
# """
# # Check the shape of points
# if points.ndim != 2 or points.shape[1] != 3:
# raise ValueError("Invalid shape of points")

# if points.shape[0] < 1:
# return np.empty((0, 2)), np.empty((0,))

# # Transform points to the camera coordinate system
# points_homo = np.column_stack((points, np.ones(points.shape[0])))
# camera_points = points_homo @ extrinsic[:3, :].T

# # Filter out points with negative depth
# mask_visible = camera_points[:, 2] > 0

# # Project Points
# # TODO: fix weird projection with distortion
# image_points, _ = cv2.projectPoints(
# camera_points, np.zeros(3), np.zeros(3), intrinsic, None
# )
# image_points = image_points.squeeze().reshape(-1, 2)

# # Replace non-projectable points and out-of-bounds points with nan
# image_points[~mask_visible, :] = np.nan

# # Filter out points falling outside the image boundaries
# if image_size is not None:
# mask_in_image = (
# (image_points[:, 0] >= 0)
# & (image_points[:, 0] < image_size[0])
# & (image_points[:, 1] >= 0)
# & (image_points[:, 1] < image_size[1])
# )
# image_points[~mask_in_image, :] = np.nan
# depths = camera_points[:, 2]

# # Remove NaN
# if not keep_size:
# mask_nan = np.isnan(image_points[:, 0])
# image_points = image_points[~mask_nan, :]
# depths = depths[~mask_nan]

# return image_points, depths


# def project_bbox_3d_to_2d(
# bbox_3d: Bbox3D,
# extrinsic: np.ndarray,
# intrinsic: np.ndarray,
# image_size: np.ndarray,
# dist_coeff: Optional[np.ndarray] = None,
# ) -> Tuple[Bbox2D, float]:
# """
# Project 3D bounding box onto the image plane. If the bounding box is
# behind the camera, compute the intersection of the bounding box with the
# camera x-y plane and project the intersection points onto the image plane.

# Args:
# bbox_3d: (cX, cY, cZ, l, w, h, roll, pitch, yaw) of the bounding box
# extrinsic: (4, 4) extrinsic matrix
# intrinsic: (3, 3) intrinsic matrix
# image_size: (2, ) image size (width, height) in pixels
# dist_coeff: (5, ) dist_coeff coefficients (k1, k2, p1, p2, k3)

# Returns:
# x1, y1, x2, y2: coordinates of the bounding box (in pixels)
# occlusion_ratio: cropped ratio of 2d bbox by image boundary
# """
# # Get corners of the bounding box in the LiDAR coordinate system
# corners, edges = get_corners_3d_bbox(*bbox_3d)

# # Project corners on image
# corners_image, _ = project_points_3d_to_2d(
# corners, extrinsic, intrinsic, image_size, dist_coeff, True
# )

# # Corner Validity
# corners_validity = np.logical_not(np.isnan(corners_image[:, 0]))

# # When all corners are behind the camera, corners_image is empty
# if np.count_nonzero(corners_validity) == 0:
# return (None, None, None, None), 1.0

# # when all corners are availble to project
# if np.count_nonzero(corners_validity) == 8:
# x1 = int(corners_image[:, 0].min())
# y1 = int(corners_image[:, 1].min())
# x2 = int(corners_image[:, 0].max())
# y2 = int(corners_image[:, 1].max())
# return (x1, y1, x2, y2), 0.0

# # When some corners are outside the image boundaries, compute the
# # intersection of edges of the bounding box with the image boundaries
# from helpers.image_utils import (
# compute_bbox_area,
# crop_2d_bbox,
# clip_line_with_image_size,
# )

# corners_homo = np.column_stack((corners, np.ones(corners.shape[0])))
# corners_camera = corners_homo @ extrinsic[:3, :].T

# corners_outside = np.empty((0, 2))
# for p1_idx, p2_idx in edges:
# # Skip if both corners are inside or outside the image
# if corners_validity[p1_idx] == corners_validity[p2_idx]:
# continue
# # Compute the intersection with xy plane in Camera Frame
# # when one corner is behind the camera
# p1_camera = corners_camera[p1_idx]
# p2_camera = corners_camera[p2_idx]
# intersection = line_segment_intersection_plane(
# p1_camera, p2_camera, (0, 0, 1, 1)
# )
# # Replace behind corners with intersection
# if intersection is not None:
# if p1_camera[2] < 0:
# p1_camera = intersection
# if p2_camera[2] < 0:
# p2_camera = intersection
# # Project the corners onto the image plane and clip with the image
# p1_image, _ = cv2.projectPoints(
# p1_camera, np.zeros(3), np.zeros(3), intrinsic, dist_coeff
# )
# p2_image, _ = cv2.projectPoints(
# p2_camera, np.zeros(3), np.zeros(3), intrinsic, dist_coeff
# )
# p1_image = p1_image.squeeze()
# p2_image = p2_image.squeeze()

# # Clip the line with the image boundaries
# p1, p2 = clip_line_with_image_size(p1_image, p2_image, image_size)
# if p1 is None or p2 is None:
# continue
# corners_image = np.vstack((corners_image, p1, p2))
# if not np.isclose(p1_image, p1).all():
# corners_outside = np.vstack((corners_outside, p1_image, p1))
# if not np.isclose(p2_image, p2).all():
# corners_outside = np.vstack((corners_outside, p2_image, p2))

# # Remove NaN
# corners_image = corners_image[~np.isnan(corners_image[:, 0])]
# corners_outside = corners_outside[~np.isnan(corners_outside[:, 0])]

# # Get valid 2d bounding box
# x1 = int(corners_image[:, 0].min())
# y1 = int(corners_image[:, 1].min())
# x2 = int(corners_image[:, 0].max())
# y2 = int(corners_image[:, 1].max())
# valid_bbox_2d = (x1, y1, x2, y2)

# if len(corners_outside) < 1:
# return valid_bbox_2d, 0.0

# # Get invalid 2d bounding box
# x1_outside = int(corners_outside[:, 0].min())
# y1_outside = int(corners_outside[:, 1].min())
# x2_outside = int(corners_outside[:, 0].max())
# y2_outside = int(corners_outside[:, 1].max())
# invalid_bbox_2d = (x1_outside, y1_outside, x2_outside, y2_outside)

# # Compute occlusion ratio
# valid_bbox_2d_area = compute_bbox_area(valid_bbox_2d)
# invalid_bbox_2d_area = compute_bbox_area(invalid_bbox_2d)
# bbox_2d_area = valid_bbox_2d_area + invalid_bbox_2d_area
# occlusion_ratio = invalid_bbox_2d_area / bbox_2d_area

# return valid_bbox_2d, occlusion_ratio


# def line_segment_intersection_2d(
# p1: Tuple[float, float],
# p2: Tuple[float, float],
# q1: Tuple[float, float],
# q2: Tuple[float, float],
# ) -> Tuple[float, float]:
# """
# Calculate the intersection of two line segments in 2D.

# Args:
# p1, p2: end-points of the first line
# q1, q2: end-points of the second line

# Returns:
# x, y: coordinates of the intersection point
# """
# # ax + by + c = 0
# a1 = p1[1] - p2[1]
# b1 = p2[0] - p1[0]
# c1 = -(b1 * p1[1] + a1 * p1[0])

# a2 = q1[1] - q2[1]
# b2 = q2[0] - q1[0]
# c2 = -(b2 * q1[1] + a2 * q1[0])

# # Calculate the intersection point
# x, y = line_intersection_2d((a1, b1, c1), (a2, b2, c2))
# # Check if the intersection point exists
# if x is None or y is None:
# return None, None
# # Check if the intersection point is on the line segment
# p_min_x = min(p1[0], p2[0])
# p_max_x = max(p1[0], p2[0])
# p_min_y = min(p1[1], p2[1])
# p_max_y = max(p1[1], p2[1])
# q_min_x = min(q1[0], q2[0])
# q_max_x = max(q1[0], q2[0])
# q_min_y = min(q1[1], q2[1])
# q_max_y = max(q1[1], q2[1])
# if (
# (not np.isclose(x, p_min_x, atol=1e-6) and x < p_min_x)
# or (not np.isclose(x, p_max_x, atol=1e-6) and x > p_max_x)
# or (not np.isclose(y, p_min_y, atol=1e-6) and y < p_min_y)
# or (not np.isclose(y, p_max_y, atol=1e-6) and y > p_max_y)
# or (not np.isclose(x, q_min_x, atol=1e-6) and x < q_min_x)
# or (not np.isclose(x, q_max_x, atol=1e-6) and x > q_max_x)
# or (not np.isclose(y, q_min_y, atol=1e-6) and y < q_min_y)
# or (not np.isclose(y, q_max_y, atol=1e-6) and y > q_max_y)
# ):
# return None, None

# return x, y


# def line_intersection_2d(
# line_coeff1: Tuple[float, float, float], line_coeff2: Tuple[float, float, float]
# ) -> Tuple[float, float]:
# """
# Calculate the intersection of two lines in 2D.

# Args:
# coeff1: (a, b, c) coefficients of the first line  (ax + by + c = 0)
# coeff2: (a, b, c) coefficients of the second line (ax + by + c = 0)

# Returns:
# x, y: coordinates of the intersection point
# """
# # ax + by + c = 0
# a1, b1, c1 = line_coeff1
# a2, b2, c2 = line_coeff2

# # Calculate the intersection point
# denom = a1 * b2 - a2 * b1
# if denom == 0:
# return None, None
# x = (b1 * c2 - b2 * c1) / denom
# y = (a2 * c1 - a1 * c2) / denom

# return x, y


# def line_segment_intersection_plane(
# p1: np.ndarray, p2: np.ndarray, plane_coeff: Tuple[float, float, float, float]
# ) -> Optional[np.ndarray]:
# """
# Calculate the intersection of a line segment with a plane.
# The line segment is defined by two end-points p1 and p2.

# Args:
# p1: end-point on the line (3,)
# p2: another end-point on the line (3,)
# plane_coeff: coefficient of the plane (ax + by + cz + d = 0)
# Returns:
# intersection point (3,)
# """
# a, b, c, d = plane_coeff
# p1p2 = p2 - p1
# # Calculate the intersection point
# denom = a * p1p2[0] + b * p1p2[1] + c * p1p2[2]
# if denom == 0:
# return None
# t = -(a * p1[0] + b * p1[1] + c * p1[2] + d) / denom
# intersection = p1 + t * p1p2

# # Check if the intersection point is on the line segment
# if t < 0 or t > 1:
# return None

# return intersection


# def transform_plane(
# plane_coeff: Tuple[float, float, float, float], extrinsic: np.ndarray
# ) -> Tuple[float, float, float, float]:
# """
# Transform a plane with the given extrinsic matrix (LiDAR to camera).

# Args:
# plane_coeff: coefficients of the plane (ax + by + cZ + d = 0)
# extrinsic: (4, 4) extrinsic matrix

# Returns:
# transformed_plane_coeff: coefficients of the transformed plane
# """
# normal = np.array([a, b, c, 0]).reshape(4, 1)
# transformed_normal = (extrinsic @ normal)[:3]

# point = np.array([0, 0, -d / c, 1]).reshape(4, 1)
# transformed_point = (extrinsic @ point)[:3]

# transformed_a = transformed_normal[0].item()
# transformed_b = transformed_normal[1].item()
# transformed_c = transformed_normal[2].item()
# transformed_d = (-transformed_normal.T @ transformed_point).item()

# return transformed_a, transformed_b, transformed_c, transformed_d


# def filter_points_inside_3d_bbox(points: np.ndarray, bbox_3d: Bbox3D) -> np.ndarray:
# """
# Filter out points that are inside the 3D bounding box

# Args:
# points: (N, 3) array of 3D points
# bbox_3d: (cX, cY, cZ, l, w, h, roll, pitch, yaw) of the bounding box

# Returns:
# points: (M, 3) array of 3D points that are inside the bounding box
# """
# cX, cY, cZ, l, w, h, roll, pitch, yaw = bbox_3d

# # Transformation matrix from the given roll, pitch, yaw
# rot_mat = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_matrix()

# # Transform points to the local frame
# transformed_points = (points - np.array([cX, cY, cZ])) @ rot_mat

# # Filter out points that are outside the bounding box
# mask_inside = (
# (transformed_points[:, 0] >= -l / 2)
# & (transformed_points[:, 0] <= l / 2)
# & (transformed_points[:, 1] >= -w / 2)
# & (transformed_points[:, 1] <= w / 2)
# & (transformed_points[:, 2] >= -h / 2)
# & (transformed_points[:, 2] <= h / 2)
# )

# return points[mask_inside]
