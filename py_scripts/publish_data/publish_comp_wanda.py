"""
Author:      Dongmyeong Lee (domlee[at]utexas[dot]edu)
Date:        Apr 24, 2024
Description: Publish compensated data for Wanda
"""

import os
import sys
import pathlib
import argparse
from tqdm import tqdm
import time
from natsort import natsorted
import pathlib

import numpy as np
import rospy

import tf2_ros
from sensor_msgs.msg import PointCloud2
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry, Path

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
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


def main(args):
    print("Publishing compensated data for CODa...")

    rospy.set_param("use_sim_time", True)
    rospy.init_node("Wanda_compenstaed_data_publisher", anonymous=True)

    # Data Publishers
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=10)
    pc_pub = rospy.Publisher(PC_TOPIC, PointCloud2, queue_size=10)
    # Odometry & Path Publisher
    odom_pub = rospy.Publisher("/odom", Odometry, queue_size=1)
    path_pub = rospy.Publisher("/global_path", Path, queue_size=1)
    global_path = Path()
    global_path.header.frame_id = GLOBAL_FRAME
    # Define TF Broadcaster
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    # Start-up delay
    wait_for_subscribers([pc_pub, odom_pub])

    # Load the poses and pointclouds
    pose_np = np.loadtxt(args.pose_file)[:, :8]
    pc_files = natsorted(os.listdir(args.pc_dir))
    assert len(pose_np) == len(pc_files), f"{len(pose_np)} != {len(pc_files)}"

    # Main Loop
    last_time = time.time()
    for idx, pose in tqdm(enumerate(pose_np), total=len(pose_np)):
        if rospy.is_shutdown():
            break

        ts = rospy.Time.from_sec(pose[0])
        clock_pub.publish(Clock(ts))

        # Publish LiDAR Pose
        odom_msg = odometry_from_xyz_quat(
            pose[1:4], pose[4:8], GLOBAL_FRAME, LIDAR_FRAME, ts
        )
        odom_pub.publish(odom_msg)

        # Publish Path
        pose_msg = pose_stamped_from_xyz_quat(pose[1:4], pose[4:8], GLOBAL_FRAME, ts)
        global_path.poses.append(pose_msg)
        path_pub.publish(global_path)

        # Publish TF
        tf_msg = tf_msg_from_quat(pose[1:4], pose[4:8], GLOBAL_FRAME, LIDAR_FRAME, ts)
        tf_broadcaster.sendTransform(tf_msg)

        # Publish Pointcloud
        pc_file = args.pc_dir / pc_files[idx]
        if pc_file.exists():
            pc_np = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 3)
            pc_msg = np_to_pointcloud2(pc_np, "x y z", LIDAR_FRAME, ts)
            pc_pub.publish(pc_msg)

        # Wait for the next frame
        while args.rate > 0 and time.time() - last_time < 1.0 / args.rate:
            time.sleep(0.01)
        last_time = time.time()


def get_args():
    parser = argparse.ArgumentParser(description="Publish compensated data for Wanda")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/dongmyeong/Projects/datasets/SARA",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="gq_appld_south_tour_01_2024-03-14-10-08-34",
        help="Scene name",
    )
    parser.add_argument("--rate", type=float, default=30, help="Publishing rate")
    args = parser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset_dir)
    args.pose_file = args.dataset_dir / "poses" / "os1" / f"{args.scene}.txt"
    args.pc_dir = args.dataset_dir / "3d_comp" / f"{args.scene}"
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
