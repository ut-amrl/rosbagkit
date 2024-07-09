"""
Author: Dongmyeong Lee (domlee[at]utexas.edu)
Date:   May 5, 2024
Description: Publishes raw pointcloud and IMU data for CODa asynchronously
"""

import pathlib
import argparse
from natsort import natsorted
import threading

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, Imu
from rosgraph_msgs.msg import Clock

from thread_modules import SharedClock, publish_clock, publish_imu, publish_pointcloud


def main(args):
    print(f"Publishing raw data of {args.scene} for {args.dataset}...")
    print("NOTE: time unit is in nano-seconds")

    rospy.set_param("use_sim_time", True)
    rospy.init_node("CODa_raw_data_publisher")

    # Data Publishers
    pc_pub = rospy.Publisher(args.pc_topic, PointCloud2, queue_size=10)
    imu_pub = rospy.Publisher(args.imu_topic, Imu, queue_size=2000)
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=10)

    # Timestamps
    timestamps = np.loadtxt(args.timestamp_file, dtype=np.float64)
    print(f"Total {len(timestamps)} timestamps")

    # IMU Data
    imu_data = np.fromfile(args.imu_file, sep=" ").reshape(-1, 11)
    print(f"Total {len(imu_data)} IMU data")

    # Point Cloud Data
    pc_files = natsorted(args.pc_dir.glob("*.bin"))
    print(f"Total {len(pc_files)} pointcloud files")
    assert len(timestamps) == len(pc_files), f"{len(timestamps)} != {len(pc_files)}"

    shared_clock = SharedClock()

    clock_thread = threading.Thread(
        target=publish_clock, args=(clock_pub, shared_clock, timestamps, args.rate)
    )
    imu_thread = threading.Thread(
        target=publish_imu, args=(imu_pub, imu_data, shared_clock)
    )
    pc_thread = threading.Thread(
        target=publish_pointcloud,
        args=(pc_pub, pc_files, timestamps, args.pc_frame, shared_clock, False),
    )

    clock_thread.start()
    pc_thread.start()
    imu_thread.start()

    clock_thread.join()
    pc_thread.join()
    imu_thread.join()


def get_args():
    parser = argparse.ArgumentParser(description="Publish raw data")
    parser.add_argument(
        "--dataset", type=str, default="CODa", help="Path to the dataset"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/dongmyeong/Projects/datasets/CODa",
        help="Path to the dataset",
    )
    parser.add_argument("--scene", type=str, help="Scene name")
    parser.add_argument("--pc_frame", type=str, help="PC frame")
    parser.add_argument("--pc_topic", type=str, help="PC topic")
    parser.add_argument("--imu_topic", type=str, help="IMU topic")
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
        args.timestamp_file = args.dataset_dir / "timestamps" / f"{args.scene}.txt"
        args.pc_dir = args.dataset_dir / "3d_raw" / "os1" / str(args.scene)
        args.imu_file = args.dataset_dir / "poses" / "imu" / f"{args.scene}.txt"
    else:
        raise NotImplementedError(f"{args.dataset} is not implemented")

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
