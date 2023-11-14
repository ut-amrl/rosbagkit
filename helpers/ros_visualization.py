"""
Author:      Dongmyeong Lee (domlee[at]utexas.edu)
Date:        September 16, 2023
Description: ROS visualization helper functions
"""
from typing import Tuple

import rospy
import numpy as np
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

from helpers.geometry import get_corners_3d_bbox

Bbox_3D = Tuple[float, float, float, float, float, float, float, float, float]


def clear_marker_array(publisher: rospy.Publisher) -> None:
    """
    Clear all markers from the publisher's topic

    Args:
        publisher: a publisher to a marker topic

    Returns:
        None
    """
    delete_all_marker = Marker(id=0, ns="delete_markerarray", action=Marker.DELETEALL)
    publisher.publish(MarkerArray([delete_all_marker]))


def create_3d_bbox_marker(
    bbox_3d: Bbox_3D,
    frame_id: str,
    time_stamp: rospy.Time,
    namespace: str = "",
    marker_id: int = 0,
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    scale: float = 0.1,
) -> Marker:
    """
    Create a 3D bounding box marker

    Args:
        bbox_3d: 3D bounding box (cx, cy, cz, l, w, h, roll, pitch, yaw)
        frame_id: frame id of header msgs
        time_stamp: time stamp of header msgs
        namespace: namespace of the bounding box
        marker_id: id of the bounding box
        color: RGBA color of the bounding box
        scale: scale of line width

    Returns:
        marker: a 3D bounding box marker
    """
    # Create the LineList marker
    marker = Marker()
    marker.header.frame_id = frame_id
    if type(time_stamp) is not rospy.Time:
        time_stamp = rospy.Time.from_sec(time_stamp)
    marker.header.stamp = time_stamp
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    marker.scale.x = scale

    marker.pose.orientation.w = 1.0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0

    # Get 3D bounding box corners
    corners, edges = get_corners_3d_bbox(*bbox_3d)

    for start, end in edges:
        start_pt = Point(*corners[start, :])
        end_pt = Point(*corners[end, :])
        marker.points.extend([start_pt, end_pt])

    return marker


def create_text_marker(
    text: str,
    coords: Tuple[float, float, float],
    frame_id: str,
    time_stamp: rospy.Time,
    namespace: str = "",
    marker_id: int = 0,
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    scale: float = 1.0,
) -> Marker:
    """
    Create a text marker

    Args:
        text: text to be displayed
        coords: 3D coordinates of the text
        frame_id: frame id of header msgs
        time_stamp: time stamp of header msgs
        namespace: namespace of the text
        marker_id: id of the text
        color: RGBA color of the text
        scale: scale of the text (0 ~ 1)

    Returns:
        marker: a text marker
    """
    marker = Marker()
    marker.header.frame_id = frame_id
    if type(time_stamp) is not rospy.Time:
        time_stamp = rospy.Time.from_sec(time_stamp)
    marker.header.stamp = time_stamp
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.TEXT_VIEW_FACING
    marker.text = text

    marker.pose.position.x = coords[0]
    marker.pose.position.y = coords[1]
    marker.pose.position.z = coords[2]

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    marker.scale.z = scale

    return marker
