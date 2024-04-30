"""
Author:      Dongmyeong (domlee[at]utexas.edu)
Date:        Sep 23, 2023
Description: A collection of image utility functions.
"""

import os
from typing import Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch

from .visualization import (
    visualize_normalized_image,
    visualize_rgbd_image,
    draw_points_on_image,
)


def get_disparity_map(img_left, img_right):
    assert img_left.shape == img_right.shape

    # SGBM Parameters
    block_size = 5
    C = img_left.shape[2] if len(img_left.shape) == 3 else 1  # RGB or Grayscale

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=block_size * 16,  # max_disp has to be dividable by 16
        blockSize=block_size,
        P1=8 * C * block_size**2,
        P2=32 * C * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        # mode=cv2.StereoSGBM_MODE_HH,
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # WLSFilter Parameters
    lmbda = 80000
    sigma = 1.2
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    # compute disparity maps
    disp_left = left_matcher.compute(img_left, img_right)
    disp_right = right_matcher.compute(img_right, img_left)

    visualize_rgbd_image(img_left, disp_left)

    # apply wls filter
    filtered_left = wls_filter.filter(disp_left, img_left, None, disp_right)
    filtered_right = wls_filter.filter(-disp_right, img_right, None, disp_left)

    # visualize_rgbd_image(img_left, filtered_left)


def draw_epipolar_lines(img1, img2):
    if img1 is None or img2 is None:
        raise ValueError("Failed to load the images")

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Ratio test as per Lowe's paper
    good = []
    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # Keep only epipolar lines
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    height = max(img1.shape[0], img2.shape[0])
    width = img1.shape[1] + img2.shape[1]
    new_img = np.zeros((height, width, 3), dtype=np.uint8)
    new_img[: img1.shape[0], : img1.shape[1]] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    new_img[: img2.shape[0], img1.shape[1] :] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2) + img1.shape[1], int(y2))
        cv2.circle(new_img, pt1, 5, color, -1)
        cv2.circle(new_img, pt2, 5, color, -1)
        cv2.line(new_img, pt1, pt2, color, 2)

    plt.imshow(new_img)
    plt.show()


def compute_overlap(
    bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int]
) -> int:
    """
    Compute the overlapping area of two bounding boxes.

    Args:
        bbox1: A tuple of (x1, y1, x2, y2) representing
               the top-left and bottom-right of the first bbox.
        bbox2: A tuple of (x1, y1, x2, y2) representing
               the top-left and bottom-right of the second bbox.

    Returns:
        The overlapping area of the two bounding boxes.
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    overlap_width = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    overlap_height = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))

    return overlap_width * overlap_height


def compute_iou(
    bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int]
) -> float:
    """
    Compute the intersection over union (IoU) of two bounding boxes.

    Args:
        bbox1: A tuple of (x1, y1, x2, y2) representing
               the top-left and bottom-right of the first bbox.
        bbox2: A tuple of (x1, y1, x2, y2) representing
               the top-left and bottom-right of the second bbox.

    Returns:
        The IoU of the two bounding boxes.
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    overlap_area = compute_overlap(bbox1, bbox2)

    return overlap_area / (bbox1_area + bbox2_area - overlap_area)


def ratio_within_image(
    bbox: tuple[int, int, int, int], image_size: tuple[int, int]
) -> float:
    """
    Compute the ratio of the bbox within the image.

    Args:
        bbox: A tuple of (x1, y1, x2, y2) representing
              the top-left and bottom-right of the bbox.
        image_size: A tuple of (width, height) representing the image size.

    Returns:
        The ratio of the bbox within the image.
    """
    x1, y1, x2, y2 = map(float, bbox)
    width, height = map(float, image_size)

    overlap_area = compute_overlap(bbox, (0, 0, width, height))
    bbox_area = (x2 - x1) * (y2 - y1)

    return overlap_area / bbox_area if not np.isclose(bbox_area, 0) else 0


def save_cropped_image_with_margin(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    target_size: tuple[int, int],
    margin_ratio: float,
    output_file: str,
) -> np.ndarray:
    """
    Save a cropped image with a margin in square shape.

    Args:
        image: The original image.
        bbox: A tuple of (x1, y1, x2, y2) representing
              the top-left and bottom-right of the bbox.
        target_size: The target size of the cropped image.
        margin_ratio: The margin ratio.
        output_file: The output file path.
    """
    # Create output directory if it does not exist.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    x1, y1, x2, y2 = map(int, bbox)
    width, height = image.shape[1], image.shape[0]

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    side_length = int(max(x2 - x1, y2 - y1) * (1 + margin_ratio))

    new_x1 = max(0, center_x - side_length // 2)
    new_y1 = max(0, center_y - side_length // 2)
    new_x2 = min(width, new_x1 + side_length)
    new_y2 = min(height, new_y1 + side_length)
    if new_x1 == 0:
        new_x2 = new_x1 + side_length
    if new_x2 == width:
        new_x1 = new_x2 - side_length

    # Check if new_y1 or new_y2 are clipped and adjust accordingly
    if new_y1 == 0:
        new_y2 = new_y1 + side_length
    if new_y2 == height:
        new_y1 = new_y2 - side_length

    # Ensure the adjusted values do not exceed image boundaries
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(width, new_x2)
    new_y2 = min(height, new_y2)

    # Crop the image
    cropped_image = image[new_y1:new_y2, new_x1:new_x2].copy()
    cv2.rectangle(
        cropped_image,
        (x1 - new_x1, y1 - new_y1),
        (x2 - new_x1, y2 - new_y1),
        (0, 255, 0),
        4,
    )
    resized_cropped_image = cv2.resize(cropped_image, target_size)
    if cropped_image.size > 0:
        cv2.imwrite(output_file, resized_cropped_image)

    return resized_cropped_image
