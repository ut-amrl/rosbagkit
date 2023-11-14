"""
Author:      Dongmyeong Lee (domlee[at]utexas[dot]edu)
Date:        September 16, 2023
Description: Convert 3d object detection results to 2d object detection results
"""
import numpy as np
from scipy.spatial import ConvexHull

from typing import Tuple, List, Optional

from helpers.geometry import (project_points_3d_to_2d, project_bbox_3d_to_2d)
from helpers.image_utils import (compute_overlap, compute_bbox_area,
                                 crop_2d_bbox)

Bbox_3d = Tuple[float, float, float, float, float, float, float, float, float]
Bbox_2d = Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]


def get_2d_bbox_from_3d_points(
    object_3d_points: np.ndarray,
    extrinsic: np.ndarray, intrinsic: np.ndarray,
    image_shape: np.ndarray, dist_coeff: np.ndarray = None
) -> Tuple[Bbox_2d, float]:
    """
    Get 2d bounding box from projected 3d points.
    args:
        object_3d_points: (N, 3) 3d points of object in LiDAR frame
        extrinsic: (4, 4) extrinsic matrix (LiDAR to camera)
        intrinsic: (3, 3) intrinsic matrix (camera)
        image_shape: (2, ) shape of image (width, height)
        dist_coeff: (5, ) distortion coefficients

    return:
        x1, y1, x2, y2: 2d bounding box of object in image frame
        occlusion_ratio: cropped ratio of 2d bbox by image boundary
    """
    projected_points, _ = project_points_3d_to_2d(
        object_3d_points, extrinsic, intrinsic, None, dist_coeff, False)

    # Mask points inside image boundary
    mask_in_image = (projected_points[:, 0] >= 0) & \
                    (projected_points[:, 0] < image_shape[0]) & \
                    (projected_points[:, 1] >= 0) & \
                    (projected_points[:, 1] < image_shape[1])

    # When all points are outside image boundary
    if not mask_in_image.any():
        return (None, None, None, None), 1.0
    
    # Get valid 2d bounding box
    x1 = int(projected_points[mask_in_image, 0].min())
    y1 = int(projected_points[mask_in_image, 1].min())
    x2 = int(projected_points[mask_in_image, 0].max())
    y2 = int(projected_points[mask_in_image, 1].max())
    bbox_2d = (x1, y1, x2, y2)

    if np.sum(~mask_in_image) < 3:
        return bbox_2d, 0.0

    # Get occlusion ratio with ConvexHull
    projected_area = ConvexHull(projected_points).volume
    invalid_area = ConvexHull(projected_points[~mask_in_image]).volume
    occlusion_ratio = invalid_area / projected_area

    return bbox_2d, occlusion_ratio


def get_2d_bboxes(
    objects_centroid: List[np.ndarray],
    objects_3d_bbox: List[Bbox_3d],
    objects_points: List[np.ndarray],
    extrinsic: np.ndarray, intrinsic: np.ndarray,
    image_shape: Tuple[int, int], dist_coeff: np.ndarray,
    distance_threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get 2d bounding box from 3d bounding box and point cloud.
    If object is close to camera, use point cloud to get 2d bounding box.
    If object is far from camera, use 3d bounding box to get 2d bounding box.

    args:
        objects_centroid: [(3,)] * N list of centroid of object in LiDAR frame
        objects_3d_bbox: [(9,)] * N list of 3d bbox of object in LiDAR frame
        objects_points: [(M,3)] * N list of object's point cloud in LiDAR frame
        extrinsic: (4, 4) extrinsic matrix (LiDAR to camera)
        intrinsic: (3, 3) intrinsic matrix (camera)
        image_shape: (2, ) shape of image (width, height)
        dist_coeff: (5, ) distortion coefficients
        distance_threshold: threshold to determine whether to 
                            use 3d bbox or point cloud to get 2d bbox

    return:
        boxes_2d: (N, 4) 2d bounding box of object in image frame
                  Top-left coordinate, bottom-right coordinate
        occlusion_ratios: (N, ) occlusion ratio of each object
        centroids_image: (N, 2) centroid of object in image frame 
    """
    # Check if all objects data have same length
    if (len(objects_centroid) != len(objects_3d_bbox) or
        len(objects_centroid) != len(objects_points)):
        raise ValueError("objects data must have same length")

    # Get number of objects
    n_objs = len(objects_centroid)
    if n_objs < 1:
        return np.empty((0, 4)), np.empty((0, )), np.empty((0, 2))

    # Convert 3d centroid to homogeneous coordinate
    objects_centroid = np.row_stack(objects_centroid)
    if objects_centroid.shape[1] == 3:
        objects_centroid = np.column_stack((objects_centroid, np.ones(n_objs)))
    elif objects_centroid.shape[1] == 4:
        assert np.allclose(objects_centroid[:, 3], 1), \
                "Last column of objects_centroid must be 1"
    else:
        raise ValueError("objects_centroid must be (N, 3) or (N, 4) array")

    # Compute object's centroid in camera frame
    objects_centroid_camera = objects_centroid @ extrinsic[:3, :].T

    # get object indices sorted by distance
    distances = np.linalg.norm(objects_centroid_camera, axis=1)
    sorted_indices = np.argsort(distances)

    # 1. Get 2d bounding box for each object
    bboxes_2d = np.empty((n_objs, 4))
    occlusion_ratios = np.zeros(n_objs)
    for i in range(len(sorted_indices)):
        # Use "point cloud" to get 2d bbox when object is close
        if distances[i] < distance_threshold:
            bboxes_2d[i], occlusion_ratios[i] = get_2d_bbox_from_3d_points(
                objects_points[i], extrinsic, intrinsic, image_shape)
        # Use "3d bounding box" to get 2d bbox when object is far
        else:
            bboxes_2d[i], occlusion_ratios[i] = project_bbox_3d_to_2d(
                objects_3d_bbox[i], extrinsic, intrinsic, image_shape)

    # 2. Get occlusion ratio for each object
    for i in range(len(sorted_indices)):
        curr_idx = sorted_indices[i]
        if np.isnan(bboxes_2d[curr_idx]).any():
            continue

        for j in range(i):
            closer_idx = sorted_indices[j]
            if np.isnan(bboxes_2d[closer_idx]).any():
                continue

            overlap_area = compute_overlap(bboxes_2d[curr_idx],
                                           bboxes_2d[closer_idx])
            if overlap_area > 0:
                curr_bbox_area = compute_bbox_area(bboxes_2d[curr_idx])
                occlusion_ratios[curr_idx] += overlap_area / curr_bbox_area
    occlusion_ratios = np.clip(occlusion_ratios, 0.0, 1.0)

    # 3. Get centroid of each object in image frame
    centroids_image, _ = project_points_3d_to_2d(objects_centroid_camera,
                                                 np.eye(4), intrinsic,
                                                 image_shape, dist_coeff, True)

    return bboxes_2d, occlusion_ratios, centroids_image
