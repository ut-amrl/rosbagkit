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

GLOBAL_FRAME = "map"
LIDAR_FRAME = "os1"
PC_TOPIC = "/ouster_points"
IMG_TOPIC = "/image_raw"


def get_args():
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
        default=[1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21],
        help="Sequence ID",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0,
        help="Publishing rate (0: no delay)",
    )
    args = parser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset)
    # args.pose_dir = args.dataset_dir / "poses" / "global_keyframe"
    # args.pose_dir = args.dataset_dir / "poses" / "dense_keyframe"
    args.pose_dir = args.dataset_dir / "poses"
    args.timestamp_dir = args.dataset_dir / "timestamps"
    args.pc_dir = args.dataset_dir / "3d_comp" / "os1"
    args.img_dir = args.dataset_dir / "2d_raw"
    args.cam_list = ["cam0", "cam1"]
    return args


def main():
    args = get_args()

    rospy.set_param("use_sim_time", True)
    rospy.init_node("CODa_compenstaed_data_publisher", anonymous=True)

    # Data Publishers
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=10)
    pc_pub = rospy.Publisher(PC_TOPIC, PointCloud2, queue_size=10)
    img_pubs = {
        cam: rospy.Publisher(f"{IMG_TOPIC}/{cam}", Image, queue_size=1)
        for cam in args.cam_list
    }
    # Odometry & Path Publisher
    odom_pub = rospy.Publisher("/odom", Odometry, queue_size=1)
    path_pub = rospy.Publisher("/global_path", Path, queue_size=1)
    # Global Path Marker
    global_path = Path()
    global_path.header.frame_id = GLOBAL_FRAME
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
        pose_file = args.pose_dir / f"{sequence}.txt"
        print(pose_file)
        pose_np = np.loadtxt(pose_file)[:, :8]

        timestamp_file = args.timestamp_dir / f"{sequence}.txt"
        timestamp_np = np.loadtxt(timestamp_file)

        # Main Loop
        for pose in tqdm(pose_np[-3000:]):
            # Get frame index
            frame = np.searchsorted(timestamp_np, pose[0], side="left")

            ts = rospy.Time.from_sec(pose[0])
            clock_pub.publish(ts)

            # Publish LiDAR Pose and Path
            odom_msg = odometry_from_xyz_quat(
                pose[1:4], pose[4:8], GLOBAL_FRAME, LIDAR_FRAME, ts
            )
            odom_pub.publish(odom_msg)

            pose_msg = pose_stamped_from_xyz_quat(
                pose[1:4], pose[4:8], GLOBAL_FRAME, ts
            )
            global_path.poses.append(pose_msg)
            path_pub.publish(global_path)

            # Publish TF
            tf_msg = tf_msg_from_quat(
                pose[1:4], pose[4:8], GLOBAL_FRAME, LIDAR_FRAME, ts
            )
            tf_broadcaster.sendTransform(tf_msg)

            # Publish Pointcloud
            pc_file = (
                args.pc_dir / str(sequence) / f"3d_comp_os1_{sequence}_{frame}.bin"
            )
            if not pc_file.exists():
                continue
            pc_np = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 3)
            pc_msg = np_to_pointcloud2(pc_np, "x y z", LIDAR_FRAME, ts)
            pc_pub.publish(pc_msg)

            # Publish Images
            for cam in args.cam_list:
                img_file = str(
                    args.img_dir
                    / cam
                    / str(sequence)
                    / f"2d_raw_{cam}_{sequence}_{frame}.jpg"
                )
                img = cv2.imread(img_file)
                img_msg = CvBridge().cv2_to_imgmsg(img, "bgr8")
                img_pubs[cam].publish(img_msg)

            # Wait for the next frame
            if args.rate > 0:
                time.sleep(1.0 / args.rate)


if __name__ == "__main__":
    main()
