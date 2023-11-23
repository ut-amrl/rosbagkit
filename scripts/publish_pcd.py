"""
Author:      Dongmyeong Lee (domlee[at]utexas.edu)
Date:        November 14, 2023
Description: Publishes a point cloud from a .pcd file
"""
import argparse

import rospy
from sensor_msgs.msg import PointCloud2

from helpers.msg_converter import pcd_to_pointcloud2
from helpers.ros_utils import wait_for_subscribers


def get_parser():
    parser = argparse.ArgumentParser(description="Publishes a .pcd file")
    parser.add_argument(
        "--pcd",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa/correction/downsampled_map_0_4.pcd",
        help="Path to .pcd file",
    )
    parser.add_argument(
        "--frame_id",
        type=str,
        default="map",
        help="Frame ID of the point cloud",
    )
    return parser


def main(args):
    rospy.init_node("pcd_publisher", anonymous=True)

    pcd_pub = rospy.Publisher("/global_map", PointCloud2, queue_size=1, latch=True)

    pc_msg = pcd_to_pointcloud2(args.pcd, "x y z", args.frame_id)

    wait_for_subscribers(pcd_pub)

    pcd_pub.publish(pc_msg)

    rospy.spin()


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
