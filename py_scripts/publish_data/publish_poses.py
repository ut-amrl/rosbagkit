"""
Author:      Dongmyeong Lee (domlee[at]utexas[dot]edu)
Date:        Apr 26, 2024
Description: Publish pose
"""

import sys
import pathlib
import argparse
import threading

import numpy as np

import rospy
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry, Path

from thread_modules import (
    SharedClock,
    publish_clock,
    publish_odom,
    publish_tf,
    publish_static_map,
)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from utils.ros_utils import wait_for_subscribers

GLOBAL_FRAME = "map"
LIDAR_FRAME = "os1"
ODOM_TOPIC = "/odom"
PATH_TOPIC = "/global_path"


def main(args):
    print(f"Publishing pose data of {args.scene} for {args.dataset}...")

    rospy.set_param("use_sim_time", True)
    rospy.init_node("pose_publisher", anonymous=True)

    # Data Publishers
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=10)
    odom_pub = rospy.Publisher(ODOM_TOPIC, Odometry, queue_size=1)
    path_pub = rospy.Publisher(PATH_TOPIC, Path, queue_size=1)

    # Load Pose & Timestamp
    pose_np = np.loadtxt(args.pose_file)[:, :8]
    timestamps = pose_np[:, 0]

    # Start-up delay
    wait_for_subscribers([odom_pub])

    # Load and publish static map
    if args.map:
        print(f"Loading static map from {args.map}")
        publish_static_map(args.map, GLOBAL_FRAME)

    shared_clock = SharedClock()

    clock_thread = threading.Thread(
        target=publish_clock, args=(clock_pub, shared_clock, timestamps)
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
    odom_thread.start()
    tf_thread.start()

    clock_thread.join()
    odom_thread.join()
    tf_thread.join()


def get_args():
    parser = argparse.ArgumentParser(description="Publish pose data")
    parser.add_argument("-f", "--pose_file", type=str, required=True)
    parser.add_argument("--map", type=str, help="Static map file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
