"""
Author:      Dongmyeong Lee (domlee[at]utexas[dot]edu)
Date:        Nov 13, 2023
Description: Publish compensated data for CODa
"""
import os
import pathlib
import argparse
from tqdm import tqdm
import time

import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge

import tf2_ros
from sensor_msgs.msg import PointCloud2, Image
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry, Path

from utils.msg_converter import (
    np_to_pointcloud2,
    pose_stamped_from_xyz_quat,
    odometry_from_xyz_quat,
    tf_msg_from_quat,
)
from utils.ros_utils import wait_for_subscribers

# Frames
global_frame = "map"
lidar_frame = "base_link"
cam_list = ["cam0", "cam1"]


def get_parser():
    parser = argparse.ArgumentParser(description="Publish raw data of CODa")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa",
        help="Path to the dataset",
    )
    parser.add_argument(
        "-s",
        "--sequence",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5],
        help="Sequence ID",
    )
    parser.add_argument(
        "--pose_type",
        choices=["", "dense", "correct", "keyframe", "inekfodom", "inekfodom/sync"],
        default="",
        help="Pose type (Options: dense, correct, keyframe)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0,
        help="Publishing rate (0: no delay)",
    )
    parser.add_argument(
        "--pointcloud_topic",
        type=str,
        default="/velodyne_points",
        help="Pointcloud topic name",
    )
    parser.add_argument(
        "--img_topic",
        type=str,
        default="/image_rect",
        help="Image topic name",
    )
    return parser


def main(args):
    rospy.set_param("use_sim_time", True)
    rospy.init_node("CODa_publisher", anonymous=True)

    # Data Publishers
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=10)
    pc_pub = rospy.Publisher(args.pointcloud_topic, PointCloud2, queue_size=10)
    img_pubs = {
        cam: rospy.Publisher(f"{args.img_topic}/{cam}", Image, queue_size=1)
        for cam in cam_list
    }
    # Odometry & Path Publisher
    odom_pub = rospy.Publisher("/odom", Odometry, queue_size=1)
    path_pub = rospy.Publisher("/global_path", Path, queue_size=1)
    # Global Path Marker
    global_path = Path()
    global_path.header.frame_id = global_frame
    # Define TF Broadcaster
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    # Start-up delay
    wait_for_subscribers([pc_pub, odom_pub])

    # Data Path
    dataset_path = pathlib.Path(args.dataset)

    # Main Loop
    for sequence in args.sequence:
        print(f"Publishing sequence {sequence}...")

        # Load Pose & Timestamp
        pose_file = dataset_path / "poses" / args.pose_type / f"{sequence}.txt"
        pose_np = np.loadtxt(pose_file, delimiter=" ")[:, :8]

        timestamp_file = dataset_path / "timestamps" / f"{sequence}.txt"
        timestamp_np = np.loadtxt(timestamp_file, delimiter=" ")

        # Data Path
        pc_root_dir = dataset_path / "3d_comp" / "os1" / str(sequence)
        img_root_dirs = {
            cam: dataset_path / "2d_rect" / cam / str(sequence) for cam in cam_list
        }

        # Main Loop
        for pose in tqdm(pose_np, total=len(pose_np)):
            # Get frame index
            frame = np.searchsorted(timestamp_np, pose[0], side="left")

            # Publish Clock
            ts = rospy.Time.from_sec(pose[0])
            clock_pub.publish(ts)

            # Publish LiDAR Pose and Path
            odom_msg = odometry_from_xyz_quat(
                pose[1:4], pose[4:8], global_frame, lidar_frame, ts
            )
            odom_pub.publish(odom_msg)

            pose_msg = pose_stamped_from_xyz_quat(
                pose[1:4], pose[4:8], global_frame, ts
            )
            global_path.poses.append(pose_msg)
            path_pub.publish(global_path)

            # Publish TF
            tf_msg = tf_msg_from_quat(
                pose[1:4], pose[4:8], global_frame, lidar_frame, ts
            )
            tf_broadcaster.sendTransform(tf_msg)

            # Data Path (index of timestamp is the same as the index of frame)
            pc_file = pc_root_dir / f"3d_comp_os1_{sequence}_{frame}.bin"
            img_files = {
                cam: img_root_dirs[cam] / f"2d_rect_{cam}_{sequence}_{frame}.jpg"
                for cam in cam_list
            }

            # Publish Pointcloud
            pc_np = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)
            pc_msg = np_to_pointcloud2(pc_np, "x y z intensity", lidar_frame, ts)
            pc_pub.publish(pc_msg)

            # Publish Images
            for cam in cam_list:
                img = cv2.imread(str(img_files[cam]))
                img_msg = CvBridge().cv2_to_imgmsg(img, "bgr8")
                img_pubs[cam].publish(img_msg)

            # Wait for the next frame
            if args.rate > 0:
                time.sleep(1.0 / args.rate)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
