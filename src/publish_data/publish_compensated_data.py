"""
Author:      Dongmyeong Lee (domlee[at]utexas[dot]edu)
Date:        Nov 13, 2023
Description: Publish compensated data for CODa
"""

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

from src.utils.ros_utils import wait_for_subscribers


def main(args):
    print(f"Publishing compensated data for {args.scene} scene of {args.dataset}...")

    rospy.set_param("use_sim_time", True)
    rospy.init_node("CODa_compenstaed_data_publisher", anonymous=True)

    # Data Publishers
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=10)
    pc_pub = rospy.Publisher(args.pc_topic, PointCloud2, queue_size=100)
    odom_pub = rospy.Publisher(args.odom_topic, Odometry, queue_size=10)
    path_pub = rospy.Publisher(args.path_topic, Path, queue_size=10)

    # Load Pose & Timestamp
    pose_np = np.loadtxt(args.pose_file)[:, :8]
    pc_files = natsorted(args.pc_dir.glob("*.bin"))
    pc_timestamps = pose_np[:, 0]
    assert len(pose_np) == len(pc_files), f"{len(pose_np)} != {len(pc_files)}"
    print(f"Total {len(pose_np)} pose data loaded from {args.pose_file}")

    # Start-up delay
    wait_for_subscribers([pc_pub, odom_pub])

    shared_clock = SharedClock()

    clock_thread = threading.Thread(
        target=publish_clock,
        args=(
            clock_pub,
            shared_clock,
            pc_timestamps,
            args.rate,
        ),
    )
    pc_thread = threading.Thread(
        target=publish_pointcloud,
        args=(
            pc_pub,
            pc_files,
            pc_timestamps,
            args.pc_frame,
            shared_clock,
            True,
            args.blind,
        ),
    )
    odom_thread = threading.Thread(
        target=publish_odom,
        args=(
            odom_pub,
            path_pub,
            pose_np,
            args.origin_frame,
            args.pc_frame,
            shared_clock,
        ),
    )
    tf_thread = threading.Thread(
        target=publish_tf,
        args=(
            pose_np,
            args.origin_frame,
            args.pc_frame,
            shared_clock,
        ),
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
    parser.add_argument("--blind", type=float, default=0.0, help="Blind range")

    parser.add_argument("--origin_frame", type=str, default="map")
    parser.add_argument("--pc_frame", type=str, default="os1")
    parser.add_argument("--pc_topic", type=str, default="/ouster_points")
    parser.add_argument("--odom_topic", type=str, default="/odom")
    parser.add_argument("--path_topic", type=str, default="/global_path")
    parser.add_argument(
        "-r",
        "--rate",
        type=float,
        default=1.0,
        help="Multiply the publish rate by FACTOR",
    )
    args = parser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset_dir)
    if args.dataset == "CODa":
        args.pc_dir = args.dataset_dir / "3d_comp_new" / "os1" / f"{args.scene}"
        # args.pose_file = args.dataset_dir / "poses" / f"{args.scene}.txt"
        args.pose_file = (
            args.dataset_dir / "poses" / "ct-icp_sync" / f"{args.scene}.txt"
        )
    elif args.dataset == "Wanda":
        args.pc_dir = args.dataset_dir / "3d_comp" / args.scene
        args.pose_file = args.dataset_dir / "poses" / args.scene / "os1.txt"
    else:
        raise NotImplementedError(f"{args.dataset} is not implemented")

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
