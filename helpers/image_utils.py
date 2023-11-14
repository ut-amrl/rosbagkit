"""
Author:      Dongmyeong (domlee[at]utexas.edu)
Date:        September 23, 2023
Description: A collection of image utility functions.
"""
import os
from typing import Tuple, Optional

import numpy as np
import cv2

from helpers.geometry import line_segment_intersection_2d


def compute_bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    """
    Compute the area of a bounding box.

    Args:
        bbox: A tuple of (x1, y1, x2, y2) representing
              the top-left and bottom-right of the bbox.

    Returns:
        The area of the bounding box.
    """
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def compute_overlap(
    bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
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
    bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
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


def clip_line_with_image_size(
    p1: Tuple[int, int], p2: Tuple[int, int], image_size: Tuple[int, int]
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Clip a line with the image size.

    Args:
        p1: A tuple of (x, y) representing the first point.
        p2: A tuple of (x, y) representing the second point.
        image_size: A tuple of (width, height) representing the image size.

    Returns:
        A tuple of points (x, y) representing the clipped line.
    """
    x1, y1 = p1
    x2, y2 = p2
    width, height = image_size

    is_p1_inside = x1 >= 0 and x1 < width and y1 >= 0 and y1 < height
    is_p2_inside = x2 >= 0 and x2 < width and y2 >= 0 and y2 < height

    if is_p1_inside and is_p2_inside:
        return p1, p2

    if not is_p1_inside and not is_p2_inside:
        return (None, None)

    boundaries = [
        ((0, 0), (width, 0)),
        ((width, 0), (width, height)),
        ((width, height), (0, height)),
        ((0, height), (0, 0)),
    ]

    for boundary in boundaries:
        point = line_segment_intersection_2d(p1, p2, boundary[0], boundary[1])

        if point[0] is not None and point[1] is not None:
            return (p1, point) if is_p1_inside else (point, p2)

    return (None, None)


def crop_2d_bbox(
    bbox: Tuple[int, int, int, int], image_size: Tuple[int, int]
) -> Tuple[Tuple[int, int, int, int], float]:
    """
    Crop a 2D bounding box with the image boundary.

    Args:
        bbox: A tuple of (x1, y1, x2, y2) representing
              the top-left and bottom-right of the bbox.
        image_size: A tuple of (width, height) representing the image size.

    Returns:
        x1, y1, x2, y2: The cropped bounding box.
        cropped_ratio: The ratio of the cropped area to the original area.
    """
    x1, y1, x2, y2 = bbox
    width, height = image_size

    valid_x1 = np.clip(x1, 0, width)
    valid_y1 = np.clip(y1, 0, height)
    valid_x2 = np.clip(x2, 0, width)
    valid_y2 = np.clip(y2, 0, height)
    valid_bbox = (valid_x1, valid_y1, valid_x2, valid_y2)

    bbox_area = compute_bbox_area(bbox)
    valid_bbox_area = compute_bbox_area(valid_bbox)
    cropped_area = bbox_area - valid_bbox_area
    cropped_ratio = np.clip(cropped_area / bbox_area, 0, 1)

    return valid_bbox, cropped_ratio


def save_cropped_image_with_margin(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    target_size: Tuple[int, int],
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
