import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def read_depth_msg(msg) -> np.ndarray:
    """
    Convert a depth image message to a 2D NumPy array in float32 meters.

    Args:
        msg (sensor_msgs.Image or sensor_msgs.CompressedImage)

    Returns:
        np.ndarray: Depth image in meters (float32)
    """
    if msg.data is None or len(msg.data) == 0:
        return None

    if hasattr(msg, "format") and "compressed" in msg.format.lower():
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Failed to decode compressed depth image.")

        if img.dtype == np.uint16:
            return img.astype(np.float32) / 1000.0  # mm -> m
        elif img.dtype == np.float32:
            return img
        else:
            raise ValueError(f"Unsupported compressed depth dtype: {img.dtype}")

    encoding = getattr(msg, "encoding", "").lower()

    if encoding == "16uc1" or encoding == "mono16":
        img = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
        return img.astype(np.float32) / 1000.0  # mm -> m
    elif encoding == "32fc1":
        img = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
        return img
    else:
        raise ValueError(f"Unsupported depth encoding: {msg.encoding}")


def save_depth(depth: np.ndarray, filename: str) -> None:
    """
    Save a float32 depth image (in meters) as a 16-bit PNG in millimeters.

    Args:
        depth (np.ndarray): Depth image in meters (float32)
        filename (str): Output PNG path
    """
    if depth.dtype != np.float32:
        raise TypeError("Input depth image must be float32 (meters).")

    # Convert meters to millimeters
    depth_mm = (depth * 1000.0).round().astype(np.uint16)

    return cv2.imwrite(filename, depth_mm)
