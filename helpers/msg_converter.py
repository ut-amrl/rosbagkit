"""
Author:      Dongmyeong Lee (domlee[at]utexas.edu)
Date:        September 16, 2023
Description: A collection of functions to convert data to ROS messages
"""
from typing import Optional, Literal
import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField, Imu
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformStamped

import open3d


def np_to_pointcloud2(
    points: np.ndarray,
    point_type: str = "x y z",
    frame_id: str = "base_link",
    time_stamp: Optional[rospy.Time] = None,
) -> PointCloud2:
    """
    Convert a numpy array to a sensor_msgs.msg.PointCloud2 message

    args:
        points: (N, M) numpy array of N points with M fields
        point_type: String with space-separated field names (e.g., "x y z i")
        frame_id: Frame in which the point cloud is defined
        time_stamp: rospy.Time() stamp to be used for the message

    return:
        sensor_msgs.msg.PointCloud2 message
    """
    field_names = point_type.split()
    assert points.shape[1] == len(field_names), "Points and point_type do not match"

    # fmt: off
    field_types = {
        "x":         (np.float32, PointField.FLOAT32),
        "y":         (np.float32, PointField.FLOAT32),
        "z":         (np.float32, PointField.FLOAT32),
        "i":         (np.float32, PointField.FLOAT32),
        "intensity": (np.float32, PointField.FLOAT32),
        "ring":      (np.uint16,  PointField.UINT16),
        "t":         (np.float32, PointField.FLOAT32),
        "time":      (np.float32, PointField.FLOAT32),
        "r":         (np.uint8,   PointField.UINT8),
        "g":         (np.uint8,   PointField.UINT8),
        "b":         (np.uint8,   PointField.UINT8),
    }

    # Check the point type and adjust fields and data shape accordingly
    fields = []
    offsets = 0
    dtype = []
    for field_name in field_names:
        if field_name not in field_types:
            raise ValueError(f"Unsupported Point Type {field_name}")
        np_dtype, ros_field_type = field_types[field_name]
        fields.append(PointField(field_name, offsets, ros_field_type, 1))
        offsets += np.dtype(np_dtype).itemsize
        dtype.append((field_name, np_dtype))

    # Convert to structured array
    structured_points = np.core.records.fromarrays(points.T, dtype=dtype)

    # Define PointCloud2
    pc2_msg = PointCloud2()
    pc2_msg.header = Header()
    pc2_msg.header.frame_id = frame_id
    pc2_msg.header.stamp = time_stamp or rospy.Time.now()

    pc2_msg.height = 1
    pc2_msg.width = points.shape[0]
    pc2_msg.point_step = offsets
    pc2_msg.row_step = pc2_msg.point_step * pc2_msg.width

    pc2_msg.is_bigendian = False
    pc2_msg.is_dense = True
    pc2_msg.fields = fields
    pc2_msg.data = structured_points.tobytes()

    return pc2_msg


def pcd_to_pointcloud2(
    pcd_file: str,
    point_type: str = "x y z",
    frame_id: str = "base_link",
    time_stamp: Optional[rospy.Time] = None,
) -> PointCloud2:
    """
    Convert a PCD file to a sensor_msgs.msg.PointCloud2 message

    Args:
        pcd_file: Path to the PCD file
        frame_id: Frame in which the point cloud is defined
        time_stamp: rospy.Time() stamp to be used for the message

    Returns:
        sensor_msgs.msg.PointCloud2 message
    """
    pcd = open3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)

    return np_to_pointcloud2(points, point_type, frame_id, time_stamp)


def np_to_imu(
    imu_data: np.ndarray,
    frame_id: str = "base_link",
    time_stamp: Optional[rospy.Time] = None,
) -> Imu:
    """
    Convert a numpy array to a sensor_msgs.msg.Imu message

    Args:
        imu_data: (7,) Numpy array of imu data [lin_acc, ang_vel, quat (w, x, y, z)]
        frame_id: Frame in which the imu data is defined
    """
    assert imu_data.shape == (10,), "IMU data must be a 10D vector"

    imu_msg = Imu()
    imu_msg.header = Header()
    imu_msg.header.frame_id = frame_id
    imu_msg.header.stamp = time_stamp or rospy.Time.now()

    imu_msg.linear_acceleration.x = imu_data[0]
    imu_msg.linear_acceleration.y = imu_data[1]
    imu_msg.linear_acceleration.z = imu_data[2]

    imu_msg.angular_velocity.x = imu_data[3]
    imu_msg.angular_velocity.y = imu_data[4]
    imu_msg.angular_velocity.z = imu_data[5]

    imu_msg.orientation.w = imu_data[6]
    imu_msg.orientation.x = imu_data[7]
    imu_msg.orientation.y = imu_data[8]
    imu_msg.orientation.z = imu_data[9]

    return imu_msg


def np_to_gps(
    gps_data: np.ndarray,
    frame_id: str = "base_link",
    time_stamp: Optional[rospy.Time] = None,
) -> Point:
    """
    Convert a numpy array to a geometry_msgs.msg.Point message

    Args:
        gps_data: (3,) Numpy array of gps data [lat, lon, alt]
        frame_id: Frame in which the gps data is defined
    """
    assert gps_data.shape == (3,), "GPS data must be a 3D vector"

    gps_msg = Point()
    gps_msg.x = gps_data[0]
    gps_msg.y = gps_data[1]
    gps_msg.z = gps_data[2]

    return gps_msg


def pose_stamped_from_xyz_quat(
    position: np.ndarray,
    quaternion: np.ndarray,
    frame_id: str = "base_link",
    time_stamp: Optional[rospy.Time] = None,
    quat_order: Literal["xyzw", "wxyz"] = "wxyz",
) -> PoseStamped:
    """
    Convert a numpy array (xyz and quaternion) to a PoseStamped message

    Args:
        position: (3,) Numpy array of position data [x, y, z]
        quaternion: (4,) Numpy array of quaternion data [qw, qx, qy, qz]
        frame_id: frame id of header msgs
        time_stamp: time stamp of header msgs
        quat_order: order of quaternion (default: wxyz)

    Returns:
        pose_msg: a PoseStamped msg
    """
    assert position.shape == (3,), "Position must be a 3D vector [x, y, z]"
    assert quaternion.shape == (4,), "Quaternion must be a 4D vector [qw, qx, qy, qz]"

    # Define PoseStamped
    pose_msg = PoseStamped()
    pose_msg.header = Header()
    pose_msg.header.frame_id = frame_id
    pose_msg.header.stamp = time_stamp or rospy.Time.now()

    # Define Pose
    pose_msg.pose.position.x = position[0]
    pose_msg.pose.position.y = position[1]
    pose_msg.pose.position.z = position[2]

    if quat_order == "wxyz":
        pose_msg.pose.orientation.w = quaternion[0]
        pose_msg.pose.orientation.x = quaternion[1]
        pose_msg.pose.orientation.y = quaternion[2]
        pose_msg.pose.orientation.z = quaternion[3]
    elif quat_order == "xyzw":
        pose_msg.pose.orientation.w = quaternion[3]
        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
    else:
        raise ValueError(f"Quaternion order {quat_order} is not supported")

    return pose_msg


def pose_stamped_from_matrix(
    transformation: np.ndarray,
    frame_id: str = "base_link",
    time_stamp: Optional[rospy.Time] = None,
) -> PoseStamped:
    """
    Convert a numpy array (4x4 transformation matrix) to a PoseStamped message

    Args:
        transformation: (4, 4) transformation matrix
        frame_id: frame id of header msgs
        time_stamp: time stamp of header msgs

    Returns:
        pose_msg: a PoseStamped msg
    """
    assert transformation.shape == (4, 4), "Transformation must be a 4x4 matrix"

    # Define PoseStamped
    pose_msg = PoseStamped()
    pose_msg.header = Header()
    pose_msg.header.frame_id = frame_id
    pose_msg.header.stamp = time_stamp or rospy.Time.now()

    # Define Pose
    pose_msg.pose.position.x = transformation[0, 3]
    pose_msg.pose.position.y = transformation[1, 3]
    pose_msg.pose.position.z = transformation[2, 3]

    quaternion = R.from_matrix(transformation[:3, :3]).as_quat()
    pose_msg.pose.orientation.w = quaternion[3]
    pose_msg.pose.orientation.x = quaternion[0]
    pose_msg.pose.orientation.y = quaternion[1]
    pose_msg.pose.orientation.z = quaternion[2]

    return pose_msg


def odometry_from_xyz_quat(
    position: np.ndarray,
    quaternion: np.ndarray,
    frame_id: str = "map",
    child_frame_id: Optional[str] = None,
    time_stamp: Optional[rospy.Time] = None,
    quat_order: Literal["xyzw", "wxyz"] = "wxyz",
) -> Odometry:
    """
    Convert a numpy array (xyz and quaternion) to a nav_msgs.msg.Odometry message

    Args:
        position: (3,) Numpy array of position data [x, y, z]
        quaternion: (4,) Numpy array of quaternion data [qw, qx, qy, qz]
        frame_id: frame id of header msgs
        child_frame_id: child frame id of header msgs
        time_stamp: time stamp of header msgs
        quat_order: order of quaternion (default: wxyz)

    Returns:
        odom_msg: a nav_msgs.msg.Odometry msg
    """
    assert position.shape == (3,), "Position must be a 3D vector [x, y, z]"
    assert quaternion.shape == (4,), "Quaternion must be a 4D vector [qw, qx, qy, qz]"

    # Define Odometry
    odom_msg = Odometry()
    odom_msg.header = Header()
    odom_msg.header.frame_id = frame_id
    odom_msg.header.stamp = time_stamp or rospy.Time.now()

    if child_frame_id is not None:
        odom_msg.child_frame_id = child_frame_id

    odom_msg.pose.pose.position.x = position[0]
    odom_msg.pose.pose.position.y = position[1]
    odom_msg.pose.pose.position.z = position[2]

    if quat_order == "wxyz":
        odom_msg.pose.pose.orientation.w = quaternion[0]
        odom_msg.pose.pose.orientation.x = quaternion[1]
        odom_msg.pose.pose.orientation.y = quaternion[2]
        odom_msg.pose.pose.orientation.z = quaternion[3]
    elif quat_order == "xyzw":
        odom_msg.pose.pose.orientation.w = quaternion[3]
        odom_msg.pose.pose.orientation.x = quaternion[0]
        odom_msg.pose.pose.orientation.y = quaternion[1]
        odom_msg.pose.pose.orientation.z = quaternion[2]
    else:
        raise ValueError(f"Quaternion order {quat_order} is not supported")

    return odom_msg


def tf_msg_from_quat(
    position: np.ndarray,
    quaternion: np.ndarray,
    frame_id: str = "base_link",
    child_frame_id: str = "child_link",
    time_stamp: Optional[rospy.Time] = None,
    quat_order: Literal["xyzw", "wxyz"] = "wxyz",
) -> TransformStamped:
    """
    Convert a numpy array (xyz and quaternion) to a TFMessage message

    Args:
        position: (3,) Numpy array of position data [x, y, z]
        quaternion: (4,) Numpy array of quaternion data [qw, qx, qy, qz]
        frame_id: frame id of header msgs
        child_frame_id: child frame id of header msgs
        time_stamp: time stamp of header msgs
        quat_order: order of quaternion (default: wxyz)

    Returns:
        tf_msg: a tf2_msgs.msg.TFMessage msg
    """
    assert position.shape == (3,), "Position must be a 3D vector [x, y, z]"
    assert quaternion.shape == (4,), "Quaternion must be a 4D vector [qw, qx, qy, qz]"

    # Define TransformStamped
    tf_msg = TransformStamped()
    tf_msg.header.frame_id = frame_id
    tf_msg.child_frame_id = child_frame_id
    tf_msg.header.stamp = time_stamp or rospy.Time.now()

    # Define Transform
    tf_msg.transform.translation.x = position[0]
    tf_msg.transform.translation.y = position[1]
    tf_msg.transform.translation.z = position[2]

    if quat_order == "wxyz":
        tf_msg.transform.rotation.w = quaternion[0]
        tf_msg.transform.rotation.x = quaternion[1]
        tf_msg.transform.rotation.y = quaternion[2]
        tf_msg.transform.rotation.z = quaternion[3]
    elif quat_order == "xyzw":
        tf_msg.transform.rotation.w = quaternion[3]
        tf_msg.transform.rotation.x = quaternion[0]
        tf_msg.transform.rotation.y = quaternion[1]
        tf_msg.transform.rotation.z = quaternion[2]
    else:
        raise ValueError(f"Quaternion order {quat_order} is not supported")

    return tf_msg


def tf_msg_from_matrix(
    transformation: np.ndarray,
    frame_id: str = "base_link",
    child_frame_id: str = "child_link",
    time_stamp: Optional[rospy.Time] = None,
) -> TransformStamped:
    """
    Convert a numpy array (4x4 transformation matrix) to a TFMessage message

    Args:
        transformation: (4, 4) transformation matrix
        frame_id: frame id of header msgs
        child_frame_id: child frame id of header msgs
        time_stamp: time stamp of header msgs

    Returns:
        tf_msg: a tf2_msgs.msg.TFMessage msg
    """
    assert transformation.shape == (4, 4), "Transformation must be a 4x4 matrix"

    # Define TransformStamped
    tf_msg = TransformStamped()
    tf_msg.header.frame_id = frame_id
    tf_msg.child_frame_id = child_frame_id
    tf_msg.header.stamp = time_stamp or rospy.Time.now()

    # Define Transform
    tf_msg.transform.translation.x = transformation[0, 3]
    tf_msg.transform.translation.y = transformation[1, 3]
    tf_msg.transform.translation.z = transformation[2, 3]

    quaternion = R.from_matrix(transformation[:3, :3]).as_quat()
    pose_msg.pose.orientation.w = quaternion[3]
    pose_msg.pose.orientation.x = quaternion[0]
    pose_msg.pose.orientation.y = quaternion[1]
    pose_msg.pose.orientation.z = quaternion[2]

    return tf_msg
