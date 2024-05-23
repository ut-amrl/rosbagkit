"""
Author:      Dongmyeong Lee (domlee[at]utexas.edu)
Date:        Sep 16, 2023
Description: ROS visualization helper functions
"""

import os
import sys

import rospy
import numpy as np
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

sys.path.append(os.path.join(os.path.dirname(__file__)))
from geometry import get_3d_bbox_corners


def clear_marker_array(publisher: rospy.Publisher) -> None:
    """Clear all markers from the publisher's topic"""
    delete_all_marker = Marker()
    delete_all_marker.action = Marker.DELETEALL
    delete_marker_array = MarkerArray(markers=[delete_all_marker])
    publisher.publish(delete_marker_array)


def create_3d_bbox_marker(
    bbox: np.ndarray,
    frame_id: str,
    timestamp: rospy.Time = None,
    marker_id: int = 0,
    namespace: str = "",
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    alpha: float = 1.0,
    scale: float = 0.1,
) -> Marker:
    """
    Create a 3D bounding box marker

    Args:
        bbox: (9,) 3D bounding box (cx, cy, cz, l, w, h, roll, pitch, yaw)
        frame_id: frame id of header msgs
        timestamp: timestamp of header msgs
        namespace: namespace of the bounding box
        marker_id: id of the bounding box
        color: RGBA or RGB color of the bounding box
        scale: scale of line width

    Returns:
        marker: a 3D bounding box marker
    """
    assert bbox.shape == (9,), f"{bbox.shape} != (9,)"

    # Create the LineList marker
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = timestamp or rospy.Time.now()
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD

    # Normalize color
    if any(c > 1 for c in color):
        color = tuple(c / 255.0 for c in color)

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = alpha
    marker.scale.x = scale

    marker.pose.orientation.w = 1.0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0

    # Get 3D bounding box corners
    corners, edges = get_3d_bbox_corners(bbox)

    for start, end in edges:
        start_pt = Point(*corners[start, :])
        end_pt = Point(*corners[end, :])
        marker.points.extend([start_pt, end_pt])

    return marker


def create_filled_bbox_3d_marker(
    bbox_3d: np.ndarray,
    frame_id: str,
    timestamp: rospy.Time = None,
    marker_id: int = 0,
    namespace: str = "",
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    alpha: float = 1.0,
) -> Marker:
    """
    Create a 3D bounding box marker

    Args:
        bbox_3d: (9,) 3D bounding box (cx, cy, cz, l, w, h, roll, pitch, yaw)
        frame_id: frame id of header msgs
        timestamp: timestamp of header msgs
        namespace: namespace of the bounding box
        marker_id: id of the bounding box
        color: RGB color of the text
        alpha: alpha of the text (0 ~ 1)

    Returns:
        marker: a 3D bounding box marker
    """
    # Create the Cube marker
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = timestamp or rospy.Time.now()
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD

    # Normalize color
    if any(c > 1 for c in color):
        color = tuple(c / 255.0 for c in color)

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = alpha

    marker.pose.position.x = bbox_3d[0]
    marker.pose.position.y = bbox_3d[1]
    marker.pose.position.z = bbox_3d[2]

    quaternion = Rotation.from_euler("xyz", bbox_3d[6:9]).as_quat()
    marker.pose.orientation.x = quaternion[0]
    marker.pose.orientation.y = quaternion[1]
    marker.pose.orientation.z = quaternion[2]
    marker.pose.orientation.w = quaternion[3]

    marker.scale.x = bbox_3d[3]  # length
    marker.scale.y = bbox_3d[4]  # width
    marker.scale.z = bbox_3d[5]  # height

    return marker


def create_text_marker(
    text: str,
    coords: tuple[float, float, float],
    frame_id: str,
    timestamp: rospy.Time = None,
    namespace: str = "",
    marker_id: int = 0,
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    alpha: float = 1.0,
    scale: float = 1.0,
) -> Marker:
    """
    Create a text marker

    Args:
        text: text to be displayed
        coords: 3D coordinates of the text
        frame_id: frame id of header msgs
        timestamp: timestamp of header msgs
        namespace: namespace of the text
        marker_id: id of the text
        color: RGB color of the text
        alpha: alpha of the text (0 ~ 1)
        scale: scale of the text (0 ~ 1)

    Returns:
        marker: a text marker
    """
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = timestamp or rospy.Time.now()
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.TEXT_VIEW_FACING
    marker.text = text

    marker.pose.position.x = coords[0]
    marker.pose.position.y = coords[1]
    marker.pose.position.z = coords[2]

    # Normalize color
    if any(c > 1 for c in color):
        color = tuple(c / 255.0 for c in color)

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = alpha
    marker.scale.z = scale

    return marker


def create_ellipsoid_marker(
    ellipsoid: np.ndarray,
    frame_id: str,
    timestamp: rospy.Time = None,
    marker_id: int = 0,
    namespace: str = "",
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    alpha: float = 1.0,
    scale: float = 0.1,
) -> Marker:
    """
    Create a ellipsoid marker

    Args:
        ellipsoid: (9,) ellipsoid (cx, cy, cz, a, b, c, roll, pitch, yaw)
        frame_id: frame id of header msgs
        timestamp: timestamp of header msgs
        namespace: namespace of the ellipsoid
        marker_id: id of the ellipsoid
        color: RGBA or RGB color of the ellipsoid
        scale: scale of line width

    Returns:
        marker: a ellipsoid marker
    """
    assert ellipsoid.shape == (9,), "Ellipsoid must be (9,)"

    # Create the Sphere marker
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = timestamp or rospy.Time.now()
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD

    # Center of the ellipsoid
    marker.pose.position.x = ellipsoid[0]
    marker.pose.position.y = ellipsoid[1]
    marker.pose.position.z = ellipsoid[2]

    # Orientation of the ellipsoid
    quaternion = Rotation.from_euler("xyz", ellipsoid[6:9]).as_quat()
    marker.pose.orientation.x = quaternion[0]
    marker.pose.orientation.y = quaternion[1]
    marker.pose.orientation.z = quaternion[2]
    marker.pose.orientation.w = quaternion[3]

    # axis lengths
    marker.scale.x = ellipsoid[3]  # a
    marker.scale.y = ellipsoid[4]  # b
    marker.scale.z = ellipsoid[5]  # c

    # Normalize color
    if any(c > 1 for c in color):
        color = tuple(c / 255.0 for c in color)

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = alpha

    return marker
