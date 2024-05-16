"""
Author:      Dongmyeong Lee (domlee[at]utexas[dot]edu)
Date:        Nov 13, 2023
Description: Publish compensated data for CODa
"""

import sys
import pathlib
import argparse
from natsort import natsorted
import threading

import numpy as np
import rospy

from sensor_msgs.msg import PointCloud2
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry, Path

from thread_modules import (
    SharedClock,
    publish_clock,
    publish_pointcloud,
    publish_odom,
    publish_tf,
)

from utils.ros_utils import wait_for_subscribers

GLOBAL_FRAME = "map"
LIDAR_FRAME = "os1"
PC_TOPIC = "/ouster_points"
ODOM_TOPIC = "/odom"
PATH_TOPIC = "/global_path"


def main(args):
    print(f"Publishing compensated data of {args.scene} for {args.dataset}...")

    rospy.set_param("use_sim_time", True)
    rospy.init_node("CODa_compenstaed_data_publisher", anonymous=True)

    # Data Publishers
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=10)
    pc_pub = rospy.Publisher(PC_TOPIC, PointCloud2, queue_size=10)
    odom_pub = rospy.Publisher(ODOM_TOPIC, Odometry, queue_size=1)
    path_pub = rospy.Publisher(PATH_TOPIC, Path, queue_size=1)

    # Load Pose & Timestamp
    pose_np = np.loadtxt(args.pose_file)[:, :8]
    pc_files = natsorted(args.pc_dir.glob("*.bin"))
    timestamps = pose_np[:, 0]
    assert len(pose_np) == len(pc_files), f"{len(pose_np)} != {len(pc_files)}"

    # Start-up delay
    wait_for_subscribers([pc_pub, odom_pub])

    shared_clock = SharedClock()

    clock_thread = threading.Thread(
        target=publish_clock, args=(clock_pub, shared_clock, timestamps)
    )
    pc_thread = threading.Thread(
        target=publish_pointcloud,
        args=(pc_pub, pc_files, timestamps, LIDAR_FRAME, shared_clock, True),
    )
    odom_thread = threading.Thread(
        target=publish_odom,
        args=(odom_pub, path_pub, pose_np, GLOBAL_FRAME, LIDAR_FRAME, shared_clock),
    )
    tf_thread = threading.Thread(
        target=publish_tf,
        args=(pose_np, GLOBAL_FRAME, LIDAR_FRAME, shared_clock),
    )

    clock_thread.start()
    pc_thread.start()
    odom_thread.start()
    tf_thread.start()

    clock_thread.join()
    pc_thread.join()
    odom_thread.join()
    tf_thread.join()


def get_args():
    parser = argparse.ArgumentParser(description="Publish compensated data")
    parser.add_argument(
        "--dataset", type=str, default="CODa", choices=["CODa", "Wanda"], help="Dataset"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/dongmyeong/Projects/datasets/CODa",
        help="Path to the dataset",
    )
    parser.add_argument("--scene", type=str, help="Scene name")
    args = parser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset_dir)
    if args.dataset == "CODa":
        args.pc_dir = args.dataset_dir / "3d_comp" / "os1" / f"{args.scene}"
        args.pose_file = args.dataset_dir / "poses" / "gicp" / f"{args.scene}.txt"
    elif args.dataset == "Wanda":
        args.pc_dir = args.dataset_dir / "3d_comp" / f"{args.scene}"
        args.pose_file = args.dataset_dir / "poses" / f"{args.scene}" / "os1.txt"
    else:
        raise NotImplementedError(f"{args.dataset} is not implemented")

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
