from loguru import logger

import numpy as np
import cv2

from scipy.spatial.transform import Rotation as R

import sensor_msgs.msg
import geometry_msgs.msg


def read_image_msg(msg: sensor_msgs.msg.Image) -> np.ndarray:
    np_arr = np.frombuffer(msg.data, np.uint8)

    if hasattr(msg, "format") and "compressed" in msg.format:
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    else:
        image = np_arr.reshape(msg.height, msg.width, -1)

    return image


def read_depth_msg(msg: sensor_msgs.msg.Image) -> np.ndarray:
    # https://docs.carnegierobotics.com/S27/api.html#api:camera:depth
    np_arr = np.frombuffer(msg.data, np.float32)

    if hasattr(msg, "format") and "compressed" in msg.format:
        depth = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    else:
        depth = np_arr.reshape(msg.height, msg.width)

    return depth


def transform_to_matrix(transform_msg: geometry_msgs.msg.Transform) -> np.ndarray:
    """Convert geometry_msgs/Transform into a 4x4 transformation matrix."""
    tx, ty, tz = (
        transform_msg.translation.x,
        transform_msg.translation.y,
        transform_msg.translation.z,
    )
    qx, qy, qz, qw = (
        transform_msg.rotation.x,
        transform_msg.rotation.y,
        transform_msg.rotation.z,
        transform_msg.rotation.w,
    )
    rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [tx, ty, tz]

    return transformation_matrix
