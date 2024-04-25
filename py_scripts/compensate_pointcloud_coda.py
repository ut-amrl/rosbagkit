"""
Author: Dongmyeong Lee (domlee[at]utexas.edu)
Date:   Apr 22, 2024
Description: Compensate Pointcloud with high-frequency pose
"""

import os
import sys
import pathlib
import argparse
from tqdm import tqdm
from natsort import natsorted
import time
import matplotlib.pyplot as plt

import numpy as np

import rospy
import cv2
from cv_bridge import CvBridge

import tf2_ros
from sensor_msgs.msg import PointCloud2, Imu, Image
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry, Path

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from utils.msg_converter import np_to_pointcloud2, np_to_imu
from utils.ros_utils import start_roscore

PC_TOPIC = "/ouster_points"
ODOM_TOPIC = "/odom"


def process_pointcloud(bin_path, dt):
    N_HORIZON = 1024
    N_RING = 128

    try:
        points = np.fromfile(bin_path, dtype=np.float32).reshape(N_RING, N_HORIZON, -1)
        points = points[:, :, :3]  # x, y, z
    except IOError:
        raise IOError("Could not read the file:", bin_path)

    # time between horizontal scans
    t_values = np.linspace(0, dt, N_HORIZON, endpoint=False)
    t_expanded = np.repeat(t_values[None, :, None], N_RING, axis=0)

    # Add time information to points
    points = np.dstack((points, t_expanded))
    return points


def main(args):
    print("Compensate Pointcloud with high-frequency pose")

    rospy.init_node("Pointcloud_Compensator", anonymous=True)
    rospy.set_param("use_sim_time", True)

    # Data Publishers
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=10)
    pc_pub = rospy.Publisher(PC_TOPIC, PointCloud2, queue_size=10)
    odom_pub = rospy.Publisher(ODOM_TOPIC, Odometry, queue_size=10)
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    # Pose Data
    pose_np = np.loadtxt(args.pose_file)[:, :8]
    print(f"Total {len(pose_np)} pose data")

    # Point Cloud Data
    pc_files = natsorted(os.listdir(args.pc_dir))
    print(f"Total {len(pc_files)} pointcloud files")

    # Load Timestamp
    timestamps = np.loadtxt(args.timestamp, dtype=np.float64)
    print(f"Total {len(timestamps)} timestamps")
    assert len(timestamps) == len(pc_files)

    for frame, ts in tqdm(enumerate(timestamps)):
        if rospy.is_shutdown():
            break

        pc_begin_time = ts
        pc_end_time = (
            timestamps[frame + 1]
            if frame + 1 < len(timestamps)
            else pc_begin_time + 0.1  # 10Hz
        )
        pc_file = args.pc_dir / pc_files[frame]
        raw_points = process_pointcloud(pc_file, pc_end_time - pc_begin_time)
        print(raw_points.shape)


def get_args():
    parser = argparse.ArgumentParser(description="Compensate Pointcloud")
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset directory")
    parser.add_argument("--seq", type=int, help="Sequence number")

    args = parser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset_dir)
    args.pc_dir = args.dataset_dir / "3d_raw" / "os1" / str(args.seq)
    args.timestamp = args.dataset_dir / "timestamps" / f"{args.seq}.txt"
    args.pose_file = args.dataset_dir / "poses" / "point-lio" / f"{args.seq}.txt"

    args.out_dir = args.dataset_dir / "3d_comp" / "os1" / str(args.seq)
    os.makedirs(args.out_dir, exist_ok=True)

    return args


if __name__ == "__main__":
    roscore_process = start_roscore()
    args = get_args()
    try:
        main(args)
    finally:
        if roscore_process:
            print("Killing roscore...")
            roscore_process.kill()
