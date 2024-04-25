"""
Author: Dongmyeong Lee (domlee[at]utexas.edu)
Date:   Nov 8, 2023
Description: Publishes raw pointcloud and IMU data for CODa
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

from sensor_msgs.msg import PointCloud2, Imu, Image
from rosgraph_msgs.msg import Clock

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from utils.msg_converter import np_to_pointcloud2, np_to_imu

PC_TOPIC = "/ouster_points"
IMU_TOPIC = "/imu/data"
IMG0_TOPIC = "/camera/left/image_raw"
IMG1_TOPIC = "/camera/right/image_raw"


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


def main(args):
    print("Publishing raw data of CODa")
    print("NOTE: time unit is in nano-seconds")

    rospy.init_node("CODa_raw_data_publisher")
    rospy.set_param("use_sim_time", True)

    # Data Publishers
    pc_pub = rospy.Publisher(PC_TOPIC, PointCloud2, queue_size=10)
    img0_pub = rospy.Publisher(IMG0_TOPIC, Image, queue_size=10)
    img1_pub = rospy.Publisher(IMG1_TOPIC, Image, queue_size=10)
    imu_pub = rospy.Publisher(IMU_TOPIC, Imu, queue_size=2000)
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=10)

    # Timestamps
    timestamps = np.loadtxt(args.timestamp_file, dtype=np.float64)
    print(f"Total {len(timestamps)} timestamps")

    # Point Cloud Data
    if args.pc:
        pc_files = natsorted(os.listdir(args.pc_dir))
        print(f"Total {len(pc_files)} pointcloud files")
        assert len(timestamps) == len(pc_files)

    # Image Data
    if args.img:
        img0_files = natsorted(os.listdir(args.img0_dir))
        img1_files = natsorted(os.listdir(args.img1_dir))
        print(f"Total {len(img0_files)} image files")
        assert len(timestamps) == len(img0_files) == len(img1_files)

    # IMU Data
    if args.imu:
        imu_np = np.fromfile(args.imu_file, sep=" ").reshape(-1, 11)
        print(f"Total {len(imu_np)} IMU data")

    # Main Loop
    last_time = time.time()
    imu_last_idx = 0
    for frame, ts in tqdm(enumerate(timestamps), total=len(timestamps), miniters=1):
        if rospy.is_shutdown():
            break

        # timestamp
        timestamp = rospy.Time.from_sec(ts)
        clock_pub.publish(timestamp)

        # Publish IMU
        while args.imu and imu_last_idx < len(imu_np) and imu_np[imu_last_idx][0] < ts:
            imu_timestamp = rospy.Time.from_sec(imu_np[imu_last_idx][0])
            imu_msg = np_to_imu(imu_np[imu_last_idx][1:], "imu_link", imu_timestamp)
            imu_pub.publish(imu_msg)
            imu_last_idx += 1

        # Publish Image
        if args.img:
            img0 = cv2.imread(str(args.img0_dir / img0_files[frame]), cv2.IMREAD_COLOR)
            img0_msg = CvBridge().cv2_to_imgmsg(img0)
            img0_msg.header.stamp = timestamp
            img0_pub.publish(img0_msg)

            img1 = cv2.imread(str(args.img1_dir / img1_files[frame]), cv2.IMREAD_COLOR)
            img1_msg = CvBridge().cv2_to_imgmsg(img1)
            img1_msg.header.stamp = timestamp
            img1_pub.publish(img1_msg)

        # Publish Point Cloud
        if args.pc:
            pc_file = args.pc_dir / pc_files[frame]
            dt = timestamps[frame + 1] - ts if (frame < len(timestamps) - 1) else 0.1
            pc_msg = process_pointcloud(pc_file, dt * 1e9, "base_link", timestamp)
            pc_pub.publish(pc_msg)

        # Rate Control
        while time.time() - last_time < 1 / args.rate:
            time.sleep(0.01)
        last_time = time.time()


def get_args():
    parser = argparse.ArgumentParser(description="Publish raw data of CODa")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/dongmyeong/Projects/datasets/CODa",
        help="Path to the dataset",
    )
    parser.add_argument("--seq", type=int, default=0, help="Sequence number")
    parser.add_argument("--pc", action="store_true", help="Publish pointcloud data")
    parser.add_argument("--imu", action="store_true", help="Publish IMU data")
    parser.add_argument("--img", action="store_true", help="Publish image data")
    parser.add_argument("--rate", type=float, default=10, help="Publishing rate (Hz)")

    args = parser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset_dir)
    args.timestamp_file = args.dataset_dir / "timestamps" / f"{args.seq}.txt"
    args.pc_dir = args.dataset_dir / "3d_raw" / "os1" / str(args.seq)
    args.imu_file = args.dataset_dir / "poses" / "imu" / f"{args.seq}.txt"
    args.img0_dir = args.dataset_dir / "2d_raw" / "cam0" / str(args.seq)
    args.img1_dir = args.dataset_dir / "2d_raw" / "cam1" / str(args.seq)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
