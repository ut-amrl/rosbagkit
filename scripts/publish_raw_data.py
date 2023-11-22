"""
Author:      Dongmyeong Lee (domlee[at]utexas.edu)
Date:        November 8, 2023
Description: Publishes pointcloud and IMU data for LiDAR odometry.
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

from sensor_msgs.msg import PointCloud2, Imu, Image
from rosgraph_msgs.msg import Clock

from helpers.msg_converter import np_to_pointcloud2, np_to_imu


def get_parser():
    parser = argparse.ArgumentParser(description="Publish raw data of CODa")
    parser.add_argument(
        "-d",
        "--dataset_path",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa",
        help="Path to the dataset",
    )
    parser.add_argument("-s", "--sequence", type=int, default=0, help="Sequence number")
    parser.add_argument(
        "--pointcloud_topic",
        type=str,
        default="/velodyne_points",
        help="Pointcloud topic name",
    )
    parser.add_argument(
        "--imu_topic",
        type=str,
        default="/imu/data",
        help="IMU topic name",
    )
    parser.add_argument(
        "--img_topic",
        type=str,
        default="/image_raw",
        help="Image topic name",
    )
    return parser


def process_pointcloud(bin_path, dt, frame_id, timestamp):
    N_HORIZON = 1024
    N_RING = 128

    try:
        points = np.fromfile(bin_path, dtype=np.float32).reshape(N_RING, N_HORIZON, -1)
    except IOError:
        raise IOError("Could not read the file:", bin_path)

    # ring number
    ring_values = np.arange(N_RING)
    ring_expanded = np.repeat(ring_values[:, None, None], N_HORIZON, axis=1)

    # time between horizontal scans
    t_values = np.linspace(0, dt, N_HORIZON, endpoint=False)
    t_expanded = np.repeat(t_values[None, :, None], N_RING, axis=0)

    # Add ring and time information to points
    points = np.dstack((points, ring_expanded, t_expanded))
    points = points.reshape(-1, points.shape[-1])

    return np_to_pointcloud2(points, "x y z intensity ring time", frame_id, timestamp)


def main(args):
    rospy.set_param("use_sim_time", True)
    rospy.init_node("CODa_raw_data_publisher")

    # Define Publishers
    pc_pub = rospy.Publisher(args.pointcloud_topic, PointCloud2, queue_size=10)
    img_pub = rospy.Publisher(args.img_topic, Image, queue_size=10)
    imu_pub = rospy.Publisher(args.imu_topic, Imu, queue_size=2000)
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=10)

    # Define Paths
    dataset_path = pathlib.Path(args.dataset_path)
    sequence = args.sequence

    # Timestamps
    timestamp_file = dataset_path / "timestamps" / f"{sequence}.txt"
    timestamps = np.loadtxt(timestamp_file, dtype=np.float64)

    # Point Cloud Data
    pc_root_dir = dataset_path / "3d_raw" / "os1" / str(sequence)
    assert len(timestamps) == len(os.listdir(pc_root_dir))

    # Image Data
    img_root_dir = dataset_path / "2d_rect" / "cam0" / str(sequence)
    assert len(timestamps) == len(os.listdir(img_root_dir))

    # IMU Data
    imu_file = dataset_path / "poses" / "imu" / f"{sequence}.txt"
    imu_np = np.fromfile(imu_file, sep=" ").reshape(-1, 11)

    # Main Loop
    imu_last_idx = 0
    for frame, ts in tqdm(enumerate(timestamps), total=len(timestamps), miniters=1):
        # Publish IMU
        while imu_np[imu_last_idx][0] < ts:
            imu_timestamp = rospy.Time.from_sec(imu_np[imu_last_idx][0])
            imu_msg = np_to_imu(imu_np[imu_last_idx][1:], "imu_link", imu_timestamp)
            imu_pub.publish(imu_msg)
            imu_last_idx += 1

        # timestamp for image and pointcloud
        timestamp = rospy.Time.from_sec(ts)
        clock_pub.publish(timestamp)

        # Publish Image
        img_file = img_root_dir / f"2d_rect_cam0_{sequence}_{frame}.jpg"
        image = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        img_msg = CvBridge().cv2_to_imgmsg(image)
        img_msg.header.stamp = timestamp
        img_pub.publish(img_msg)

        # Publish Point Cloud
        dt = timestamps[frame + 1] - ts if (frame < len(timestamps) - 1) else 0.1
        pc_file = pc_root_dir / f"3d_raw_os1_{sequence}_{frame}.bin"
        pc_msg = process_pointcloud(pc_file, dt, "base_link", timestamp)
        pc_pub.publish(pc_msg)

        time.sleep(0.1) # for Sim time


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
