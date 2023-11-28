"""
Author:      Dongmyeong Lee (domlee[at]utexas[dot]edu)
Date:        November 8, 2023
Description: Publishes pointcloud and pose data from the dataset.
"""
import os
import pathlib
import argparse
from tqdm import tqdm

import numpy as np
from scipy.spatial.transform import Rotation

import rospy
import tf2_ros
import tf.transformations as tf_trans

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from helpers.msg_converter import (
    bin_to_pointcloud2,
    pose_stamped_from_quat,
    tf_msg_from_matrix,
)


def main(args):
    # Define Frames
    global_frame = "map"
    lidar_frame = "os1"

    # Define Publishers
    pc_pub = rospy.Publisher("/os1/pointcloud", PointCloud2, queue_size=1)

    # Define Pose Publishers
    lidar_pose_pub = rospy.Publisher("/os1/pose", PoseStamped, queue_size=1)
    path_pub = rospy.Publisher("/global_path", Path, queue_size=1)

    # Define Global Path Marker
    global_path = Path()
    global_path.header.frame_id = global_frame

    # Define TF Broadcaster
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    # Path to the data
    dataset_path = pathlib.Path(args.dataset_path)
    sequence = args.sequence
    pc_root_dir = dataset_path / "3d_comp" / "os1" / sequence

    # Pose DATA
    pose_file = dataset_path / "poses" / "dense" / f"{sequence}.txt"
    pose_np = np.fromfile(pose_file, sep=" ").reshape(-1, 8)

    # Main Loop
    rate = rospy.Rate(1)
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

        # Publish Point Cloud
        pc_file = pc_root_dir / f"3d_comp_os1_{sequence}_{frame}.bin"
        if os.path.exists(pc_file):
            pc_msg = bin_to_pointcloud2(pc_file, "xyzi", lidar_frame, ts)
            pc_pub.publish(pc_msg)

        rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CODa path and sequence.")
    parser.add_argument(
        "-d", "--dataset_path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "-s", "--sequence", type=str, required=True, help="Sequence number"
    )

    args = parser.parse_args()
    main(args)
