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
import warnings
import matplotlib.pyplot as plt

import numpy as np

import rospy
import cv2
from cv_bridge import CvBridge

import tf2_ros
from sensor_msgs.msg import PointCloud2, Imu, Image
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry, Path

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from utils.msg_converter import np_to_pointcloud2, np_to_imu
from utils.ros_utils import start_roscore

PC_TOPIC = "/ouster_points"
ODOM_TOPIC = "/odom"


def motion_compensation(pc_file, ts, poses):
    N_HORIZON = 1024
    N_RING = 128

    try:
        points = np.fromfile(pc_file, dtype=np.float32).reshape(N_RING, N_HORIZON, -1)
        points = points[:, :, :3]  # x, y, z
    except IOError:
        raise IOError("Could not read the file:", pc_file)

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

    # Pose Data
    dense_poses = np.loadtxt(args.dense_posefile)[:, :8]
    print(f"Total {len(dense_poses)} pose data")

    # Point Cloud Data and its Timestamp
    pc_files = natsorted(list(args.pc_dir.glob("*.bin")))
    timestamps = np.loadtxt(args.timestamp, dtype=np.float64)
    print(f"Total {len(pc_files)} pointcloud files and timestamps")
    assert len(pc_files) == len(timestamps)

    for idx, ts in tqdm(enumerate(timestamps), total=len(timestamps)):
        if rospy.is_shutdown():
            break

        # motion compensation
        comp_pc = motion_compensation(pc_files[idx], ts, dense_poses)

        # Save the compensated pointcloud
        out_file = args.out_dir / f"3d_comp_os1_{args.seq}_{idx}.bin"
        comp_pc.astype(np.float32).tofile(out_file)


def get_args():
    parser = argparse.ArgumentParser(description="Compensate Pointcloud")
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset directory")
    parser.add_argument("--seq", type=int, help="Sequence number")

    args = parser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset_dir)
    args.pc_dir = args.dataset_dir / "3d_raw" / "os1" / str(args.seq)
    args.timestamp = args.dataset_dir / "timestamps" / f"{args.seq}.txt"
    args.dense_posefile = args.dataset_dir / "poses" / "point-lio" / f"{args.seq}.txt"

    args.out_dir = args.dataset_dir / "3d_comp" / "os1" / str(args.seq)
    os.makedirs(args.out_dir, exist_ok=True)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
