"""
Author:      Dongmyeong (domlee[at]utexas.edu)
Date:        Sep 23, 2023
Description: A collection of image utility functions.
"""
import os
from typing import Optional

import numpy as np
import cv2


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
