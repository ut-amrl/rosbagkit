"""
Author:      Dongmyeong Lee (domlee[at]utexas[dot]edu)
Date:        Nov 23, 2023
Description: Get insatance 2D bounding box from 3D bounding box in global frame
"""
import os
import pathlib
import argparse
import json
from tqdm import tqdm
from typing import Tuple, List

import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN

import rospy
import cv2
from cv_bridge import CvBridge
import tf2_ros

from sensor_msgs.msg import PointCloud2, Image
from visualization_msgs.msg import Marker, MarkerArray
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry, Path

from helpers.msg_converter import (
    np_to_pointcloud2,
    pose_stamped_from_xyz_quat,
    odometry_from_xyz_quat,
    tf_msg_from_quat,
)
from helpers.geometry import transform_bbox_3d
from helpers.ros_viz_utils import create_bbox_3d_marker, clear_marker_array
from helpers.ros_utils import wait_for_subscribers
from helpers.math_utils import average_rpy

# Frames
global_frame = "map"
lidar_frame = "base_link"
cam_list = ["cam0", "cam1"]

# Classes to be detected
classes = {
    "Tree": {"id": 0, "color": (0, 1.0, 0)},
    "Pole": {"id": 1, "color": (0, 0, 1.0)},
    "Bollard": {"id": 2, "color": (1.0, 0, 0)},
}


def get_parser():
    parser = argparse.ArgumentParser(description="Get instance 2D bounding box")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa",
        help="Path to the dataset",
    )
    parser.add_argument(
        "-s",
        "--sequences",
        nargs="+",
        type=int,
        default=[0],
        help="Sequence ID",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0,
        help="Publishing rate (0: no delay)",
    )
    parser.add_argument(
        "--map_topic",
        type=str,
        default="/global_map",
        help="Global map topic name",
    )
    parser.add_argument(
        "--pointcloud_topic",
        type=str,
        default="/points",
        help="Pointcloud topic name",
    )
    parser.add_argument(
        "--image_topic",
        type=str,
        default="/image",
        help="Image topic name",
    )
    return parser


def main(args):
    rospy.set_param("use_sim_time", True)
    rospy.init_node("CODa_global_3d_bbox_getter", anonymous=True)

    # Data Publishers
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=10)
    map_pub = rospy.Publisher(args.map_topic, PointCloud2, queue_size=1, latch=True)
    pc_pub = rospy.Publisher(args.pointcloud_topic, PointCloud2, queue_size=10)
    img_pubs = {
        cam: rospy.Publisher(f"/{cam}/image_rect", Image, queue_size=1)
        for cam in cam_list
    }
    # Odometry & Path Publisher
    odom_pub = rospy.Publisher("/odom", Odometry, queue_size=1)
    path_pub = rospy.Publisher("/global_path", Path, queue_size=1)
    # Global Path Marker
    global_path = Path()
    global_path.header.frame_id = global_frame
    # Define TF Broadcaster
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    # 3D Bounding Box Publisher for each class
    object_pubs = {
        class_name: rospy.Publisher(
            f"instance_3dbbox/{class_name}", MarkerArray, queue_size=1, latch=True
        )
        for class_name in classes
    }

    # Wait for Subscribers
    wait_for_subscribers([*object_pubs.values()])

    # Data Path
    dataset_path = pathlib.Path(args.dataset)

    # Publish 3D Bounding Box
    for class_name in classes:
        bbox_3d_file = dataset_path / "3d_bbox" / "global" / f"{class_name}.json"
        if not os.path.exists(bbox_3d_file):
            continue

        bbox_3d_json = json.load(open(bbox_3d_file, "r"))

        marker_array = MarkerArray()
        for idx, bbox_3d in enumerate(bbox_3d_json["3dbbox"]):
            # fmt: off
            bbox = np.array([
                bbox_3d["cX"], bbox_3d["cY"], bbox_3d["cZ"],
                bbox_3d["h"], bbox_3d["l"], bbox_3d["w"],
                bbox_3d["r"], bbox_3d["p"], bbox_3d["y"]
            ])
            # fmt: on
            bbox_marker = create_bbox_3d_marker(
                bbox_3d=bbox,
                frame_id=global_frame,
                marker_id=idx,
                namespace=bbox_3d["instanceId"],
                color=classes[class_name]["color"],
            )
            marker_array.markers.append(bbox_marker)

        clear_marker_array(object_pubs[class_name])
        object_pubs[class_name].publish(marker_array)

    # Main Loop
    for sequence in args.sequences:
        pose_file = dataset_path / "poses" / "correct" / f"{sequence}.txt"
        pose_np = np.loadtxt(pose_file, delimiter=" ").reshape(-1, 8)

        timestamp_file = dataset_path / "timestamps" / f"{sequence}.txt"
        timestamp_np = np.loadtxt(timestamp_file, delimiter=" ")

        # Data Path
        pc_root_dir = dataset_path / "3d_comp" / "os1" / str(sequence)
        img_root_dirs = {
            cam: dataset_path / "2d_rect" / cam / str(sequence) for cam in cam_list
        }
        bbox_3d_root_dir = dataset_path / "3d_bbox" / "os1" / str(sequence)

        # Main Loop
        for pose in tqdm(pose_np, total=len(pose_np)):
            # Get Pose
            frame = np.searchsorted(timestamp_np, pose[0], side="left")
            # ts = rospy.Time.from_sec(pose[0])

            # # Publish Clock
            # clock_pub.publish(ts)

            # # Publish LiDAR Odometry and Path
            # odom_msg = odometry_from_xyz_quat(
            # pose[1:4], pose[4:], global_frame, lidar_frame, ts
            # )
            # odom_pub.publish(odom_msg)

            # pose_msg = pose_stamped_from_xyz_quat(pose[1:4], pose[4:], global_frame, ts)
            # global_path.poses.append(pose_msg)
            # path_pub.publish(global_path)

            # # Publish TF
            # tf_msg = tf_msg_from_quat(
            # pose[1:4], pose[4:], global_frame, lidar_frame, ts
            # )
            # tf_broadcaster.sendTransform(tf_msg)

            # # Data Path
            # pc_file = pc_root_dir / f"3d_comp_os1_{sequence}_{frame}.bin"
            # img_files = {
            # cam: img_root_dirs[cam] / f"2d_rect_{cam}_{sequence}_{frame}.jpg"
            # for cam in cam_list
            # }
            bbox_3d_file = bbox_3d_root_dir / f"3d_bbox_os1_{sequence}_{frame}.json"

            if not os.path.exists(bbox_3d_file):
                continue

            # # Publish Pointcloud
            # pc_np = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)
            # pc_msg = np_to_pointcloud2(pc_np, "x y z intensity", lidar_frame, ts)
            # pc_pub.publish(pc_msg)

            # # Publish Images
            # for cam in cam_list:
            # img = cv2.imread(str(img_files[cam]))
            # img_msg = CvBridge().cv2_to_imgmsg(img, "bgr8")
            # img_pubs[cam].publish(img_msg)

            # Transformation Matrix from Global to LiDAR
            H_lg = np.eye(4)
            H_lg[:3, :3] = R.from_quat(pose[[5, 6, 7, 4]]).as_matrix()
            H_lg[:3, 3] = pose[1:4]

            # # Get Averaged Bbox
            # averaged_bboxes = {
            # class_name: cluster_average_bbox_3d(bboxes[class_name])
            # for class_name in classes
            # }

            # # Publish 3D Bounding Box
            # for class_name in classes:
            # marker_array = MarkerArray()
            # for idx, bbox in enumerate(averaged_bboxes[class_name]):
            # bbox_marker = create_bbox_3d_marker(
            # bbox,
            # global_frame,
            # ts,
            # idx,
            # color=classes[class_name]["color"],
            # )
            # marker_array.markers.append(bbox_marker)
            # clear_marker_array(object_pubs[class_name])
            # object_pubs[class_name].publish(marker_array)

            # # Wait for next frame
            # if args.rate > 0:
            # time.sleep(1 / args.rate)

    rospy.spin()


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
