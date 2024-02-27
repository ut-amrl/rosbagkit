"""
Author: Dongmyeong Lee (domlee[at]utexas.edu)
Date:   Nov 8, 2023
Description: Publishes pointcloud and IMU data for LiDAR odometry.
"""
import os
import pathlib
import argparse
from tqdm import tqdm
from natsort import natsorted
import time
from termcolor import colored

import numpy as np

import rospy
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import PointCloud2, Imu, Image
from rosgraph_msgs.msg import Clock

from utils.msg_converter import np_to_pointcloud2, np_to_imu

pointcloud_topic = "/ouster_points"
imu_topic = "/imu/data"
img_topic = "/image_raw"


def get_args():
    parser = argparse.ArgumentParser(description="Publish raw data of CODa")
    parser.add_argument(
        "-d",
        "--dataset_path",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa",
        help="Path to the dataset",
    )
    parser.add_argument("-s", "--seq", type=int, default=0, help="Sequence number")
    parser.add_argument(
        "--clock",
        action="store_true",
        help="Publish clock data",
    )
    parser.add_argument(
        "--pc",
        action="store_true",
        help="Publish pointcloud data",
    )
    parser.add_argument(
        "--imu",
        action="store_true",
        help="Publish IMU data",
    )
    parser.add_argument(
        "--img",
        action="store_true",
        help="Publish image data",
    )
    parser.add_argument(
        "-r",
        "--rate",
        type=float,
        default=10,
        help="Rate of publishing data",
    )
    return parser.parse_args()


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

    # placehold for reflectivity, ambient, range
    reflectivity = np.zeros((N_RING, N_HORIZON, 1), dtype=np.float32)
    ambient = np.zeros((N_RING, N_HORIZON, 1), dtype=np.float32)
    # range is distance from the sensor to the point (mm)
    rng = np.zeros((N_RING, N_HORIZON, 1), dtype=np.float32)
    # for i in range(N_RING):
    # for j in range(N_HORIZON):
    # rng[i, j] = np.linalg.norm(points[i, j, :3])

    # Add ring and time information to points
    points = np.dstack((points, ring_expanded, t_expanded, reflectivity, ambient, rng))

    return np_to_pointcloud2(
        points, "x y z intensity ring t reflectivity ambient range", frame_id, timestamp
    )


def main():
    args = get_args()
    print(colored("Publishing raw data of CODa", "green"))
    print(colored("NOTE: time unit is in micro-seconds", "yellow"))

    rospy.set_param("use_sim_time", True)
    rospy.init_node("CODa_raw_data_publisher")

    # Define Publishers
    pc_pub = rospy.Publisher(pointcloud_topic, PointCloud2, queue_size=10)
    img_pub = rospy.Publisher(img_topic, Image, queue_size=10)
    imu_pub = rospy.Publisher(imu_topic, Imu, queue_size=2000)
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=10)

    # Define Paths
    dataset_path = pathlib.Path(args.dataset_path)
    sequence = args.seq

    # Timestamps
    timestamp_file = dataset_path / "timestamps" / f"{sequence}.txt"
    timestamps = np.loadtxt(timestamp_file, dtype=np.float64)

    # Point Cloud Data
    if args.pc:
        pc_root_dir = dataset_path / "3d_raw" / "os1" / str(sequence)
        pc_files = natsorted(os.listdir(pc_root_dir))
        # assert len(timestamps) == len(pc_files), f"{len(timestamps)} != {len(pc_files)}"

    # Image Data
    if args.img:
        img_root_dir = dataset_path / "2d_raw" / "cam0" / str(sequence)
        img2_root_dir = dataset_path / "2d_raw" / "cam1" / str(sequence)
        img_files = natsorted(os.listdir(img_root_dir))
        img2_files = natsorted(os.listdir(img2_root_dir))
        # assert len(img_files) == len(img2_files)
        # assert len(timestamps) == len(img_files)

    # IMU Data
    if args.imu:
        imu_file = dataset_path / "poses" / "imu" / f"{sequence}.txt"
        imu_np = np.fromfile(imu_file, sep=" ").reshape(-1, 11)

    # Main Loop
    imu_last_idx = 0
    for frame, ts in tqdm(enumerate(timestamps), total=len(timestamps), miniters=1):
        t1 = time.time()
        # timestamp
        timestamp = rospy.Time.from_sec(ts)
        clock_pub.publish(timestamp)

        # Publish IMU
        while args.imu and imu_np[imu_last_idx][0] < ts:
            imu_timestamp = rospy.Time.from_sec(imu_np[imu_last_idx][0])
            imu_msg = np_to_imu(imu_np[imu_last_idx][1:], "imu_link", imu_timestamp)
            imu_pub.publish(imu_msg)
            imu_last_idx += 1

        # Publish Image
        if args.img:
            image_file = img_root_dir / img_files[frame]
            image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
            img_msg = CvBridge().cv2_to_imgmsg(image)
            img_msg.header.stamp = timestamp
            img_pub.publish(img_msg)

        # Publish Point Cloud
        if args.pc:
            dt = timestamps[frame + 1] - ts if (frame < len(timestamps) - 1) else 0.1
            dt = dt * 1e6  # convert to micro-seconds
            pc_file = pc_root_dir / pc_files[frame]
            pc_msg = process_pointcloud(pc_file, dt, "base_link", timestamp)
            pc_pub.publish(pc_msg)

        t2 = time.time()
        time.sleep(max(0, 1 / args.rate - (t2 - t1)))


if __name__ == "__main__":
    main()
