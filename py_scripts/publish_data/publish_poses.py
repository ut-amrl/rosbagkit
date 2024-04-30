"""
Author:      Dongmyeong Lee (domlee[at]utexas[dot]edu)
Date:        Apr 26, 2024
Description: Publish pose
"""

import os
import sys
import pathlib
import argparse
from tqdm import tqdm
import time

import numpy as np
import rospy

import tf2_ros
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry, Path

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from utils.msg_converter import (
    pose_stamped_from_xyz_quat,
    odometry_from_xyz_quat,
    tf_msg_from_quat,
)
from utils.ros_utils import wait_for_subscribers

GLOBAL_FRAME = "map"
LIDAR_FRAME = "os1"
PC_TOPIC = "/ouster_points"
IMG0_TOPIC = "/camera/left/image_raw"
IMG1_TOPIC = "/camera/right/image_raw"


def main(args):
    rospy.set_param("use_sim_time", True)
    rospy.init_node("pose_publisher", anonymous=True)

    # Data Publishers
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=10)
    # Odometry & Path Publisher
    odom_pub = rospy.Publisher("/odom", Odometry, queue_size=1)
    path_pub = rospy.Publisher("/global_path", Path, queue_size=1)
    # Global Path Marker
    global_path = Path()
    global_path.header.frame_id = GLOBAL_FRAME
    # Define TF Broadcaster
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    # Start-up delay
    wait_for_subscribers([odom_pub])

    print(f"Publishing poses from {args.pose_file} ...")

    # Load Pose & Timestamp
    pose_np = np.loadtxt(args.pose_file)[:, :8]

    # Main Loop
    last_time = time.time()
    for idx, pose in tqdm(enumerate(pose_np), total=len(pose_np)):
        if rospy.is_shutdown():
            break

        ts = rospy.Time.from_sec(pose[0])
        clock_pub.publish(Clock(ts))

        # Publish LiDAR Pose and Path
        odom_msg = odometry_from_xyz_quat(
            pose[1:4], pose[4:8], GLOBAL_FRAME, LIDAR_FRAME, ts
        )
        odom_pub.publish(odom_msg)

        pose_msg = pose_stamped_from_xyz_quat(pose[1:4], pose[4:8], GLOBAL_FRAME, ts)
        global_path.poses.append(pose_msg)
        path_pub.publish(global_path)

        # Publish TF
        tf_msg = tf_msg_from_quat(pose[1:4], pose[4:8], GLOBAL_FRAME, LIDAR_FRAME, ts)
        tf_broadcaster.sendTransform(tf_msg)

        # Wait for the next frame
        while time.time() - last_time < 1 / args.rate:
            time.sleep(0.01)
        last_time = time.time()


def get_args():
    parser = argparse.ArgumentParser(description="Publish pose data")
    parser.add_argument("-f", "--pose_file", type=str, required=True)
    parser.add_argument(
        "--rate",
        type=float,
        default=100,
        help="Publishing rate (0: no delay)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
