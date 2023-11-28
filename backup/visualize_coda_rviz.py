"""
Author:      Dongmyeong Lee (domlee[at]utexas[dot]edu)
Date:        September 16, 2023
Description: Publish the data from CODa dataset to ROS
"""
import os
import pathlib
import argparse
import json
import sys
import termios
import tty
from tqdm import tqdm

import numpy as np
from scipy.spatial.transform import Rotation

import rospy
from cv_bridge import CvBridge
import cv2
import tf2_ros
import tf.transformations as tf_trans

from sensor_msgs.msg import PointCloud2, Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path

from helpers.msg_converter import (
    bin_to_pointcloud2,
    np_to_pointcloud2,
    pose_stamped_from_quat,
    pose_stamped_from_matrix,
    tf_msg_from_matrix,
)
from helpers.ros_visualization import (
    clear_marker_array,
    create_3d_bbox_marker,
    create_text_marker,
)
from helpers.geometry import (
    project_bbox_3d_to_2d,
    project_points_3d_to_2d,
    filter_points_inside_3d_bbox,
    get_corners_3d_bbox,
)
from object_mapper import get_2d_bboxes

from datasets.CODa.coda_utils import load_extrinsic_matrix, load_camera_params
from datasets.CODa.constants import *

# from helpers.ros_visualization import publish_3d_bbox


OCCUMULATE_FRAME = 5
DIST_THRESHOLD_PROJECT = 2
DIST_THRESHOLD_ID = 1.0
DISABLE_IOU_THRESHOLD = 0.9


def main(args):
    rospy.set_param("use_sim_time", True)
    rospy.init_node("CODa_publisher")

    # Define Frames
    global_frame = "map"
    lidar_frame = "os1"

    # Define Publishers
    pc_pub = rospy.Publisher("/os1/pointcloud", PointCloud2, queue_size=1)
    bbox_3d_pub = rospy.Publisher("/bbox_3d", MarkerArray, queue_size=1)

    cam_pubs = {
        cam: rospy.Publisher(f"/{cam}/image", Image, queue_size=1)
        for cam in ["cam0", "cam1"]
    }

    # Define Pose Publishers
    lidar_pose_pub = rospy.Publisher("/os1/pose", PoseStamped, queue_size=1)
    path_pub = rospy.Publisher("/global_path", Path, queue_size=1)

    # Object ID Publisher
    object_id_pub = rospy.Publisher("/object_id", MarkerArray, queue_size=1)

    # Define Global Path Marker
    global_path = Path()
    global_path.header.frame_id = global_frame

    # Define TF Broadcaster
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    # Path to the data
    dataset_path = pathlib.Path(args.dataset_path)
    sequence = args.sequence
    pc_root_dir = dataset_path / "3d_comp" / "os1" / sequence
    bbox_3d_root_dir = dataset_path / "3d_pred" / "os1" / sequence
    cam0_root_dir = dataset_path / "2d_rect" / "cam0" / sequence
    cam1_root_dir = dataset_path / "2d_rect" / "cam1" / sequence
    calib_dir = dataset_path / "calibrations" / sequence

    # Output Directory
    bbox_2d_out_dirs = {
        "cam0": dataset_path / "2d_proj" / "cam0" / sequence,
        "cam1": dataset_path / "2d_proj" / "cam1" / sequence,
    }
    for cam, out_dir in bbox_2d_out_dirs.items():
        os.makedirs(out_dir, exist_ok=True)

    # Pose DATA
    pose_file = dataset_path / "poses" / "dense" / f"{sequence}.txt"
    pose_np = np.fromfile(pose_file, sep=" ").reshape(-1, 8)
    lidar_ts_file = dataset_path / "timestamps" / f"{sequence}.txt"

    # Calibration DATA (Extrinsic and Intrinsic)
    os1_to_base_ext_file = calib_dir / "calib_os1_to_base.yaml"
    os1_to_cam_ext_files = {
        "cam0": calib_dir / "calib_os1_to_cam0.yaml",
        "cam1": calib_dir / "calib_os1_to_cam1.yaml",
    }
    cam_intrinsic_files = {
        "cam0": calib_dir / "calib_cam0_intrinsics.yaml",
        "cam1": calib_dir / "calib_cam1_intrinsics.yaml",
    }

    os1_to_base_ext = load_extrinsic_matrix(os1_to_base_ext_file)
    os1_to_cam_extrinsic = {
        cam: load_extrinsic_matrix(ext_file)
        for cam, ext_file in os1_to_cam_ext_files.items()
    }
    cam_intrinsics = {
        cam: load_camera_params(intrinsic_file)
        for cam, intrinsic_file in cam_intrinsic_files.items()
    }

    # Main Loop
    for sequence in range(22):
        for frame, pose in tqdm(enumerate(pose_np), total=len(pose_np), miniters=1):
            # Get Pose
            ts, x, y, z, qw, qx, qy, qz = pose
            ts = rospy.Time.from_sec(ts)

            # Get LiDAR Pose in global frame in SE(3)
            lidar_pose = np.eye(4)
            lidar_pose[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            lidar_pose[:3, 3] = [x, y, z]

            # Publish LiDAR Pose and Path
            pose_msg = pose_stamped_from_quat(pose[1:], global_frame, ts)
            global_path.poses.append(pose_msg)
            lidar_pose_pub.publish(pose_msg)
            path_pub.publish(global_path)

            # Broadcast TF (map -> os1)
            tf_msg = tf_msg_from_matrix(lidar_pose, global_frame, lidar_frame, ts)
            tf_broadcaster.sendTransform(tf_msg)

            # Get the file path to the data
            pc_file = pc_root_dir / f"3d_comp_os1_{sequence}_{frame}.bin"
            bbox_file = bbox_3d_root_dir / f"3d_bbox_os1_{sequence}_{frame}.json"
            cam_files = {
                "cam0": cam0_root_dir / f"2d_rect_cam0_{sequence}_{frame}.jpg",
                "cam1": cam1_root_dir / f"2d_rect_cam1_{sequence}_{frame}.jpg",
            }

            # Publish Point Cloud
            if os.path.exists(pc_file):
                pc_msg = bin_to_pointcloud2(pc_file, "xyzi", lidar_frame, ts)
                pc_pub.publish(pc_msg)

            # Publish Image
            for cam, cam_pub in cam_pubs.items():
                cam_file = cam_files[cam]
                if not os.path.exists(cam_file):
                    continue
                image = cv2.imread(str(cam_file), cv2.IMREAD_COLOR)
                cam_msg = CvBridge().cv2_to_imgmsg(image)
                cam_msg.header.stamp = ts
                cam_pub.publish(cam_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CODa path and sequence.")
    parser.add_argument(
        "-d", "--dataset_path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "-s", "--sequences", type=str, required=True, help="Sequence number"
    )

    args = parser.parse_args()
    main(args)
