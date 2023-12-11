"""
Author:      Dongmyeong Lee (domlee[at]utexas[dot]edu)
Date:        Nov 22, 2023
Description: Get global 3D bounding box from annotation of CODa dataset
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

# Classes to be detected
classes = {
    "Tree": {"id": 0, "color": (0, 1.0, 0)},
    "Pole": {"id": 1, "color": (0, 0, 1.0)},
    "Bollard": {"id": 2, "color": (1.0, 0, 0)},
}


def get_parser():
    parser = argparse.ArgumentParser(description="Get global 3D bounding box")
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
        default=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21],
        help="Sequence ID",
    )
    return parser


def cluster_average_bbox_3d(bboxes: np.ndarray, threshold: float = 1.5) -> np.ndarray:
    """
    Clustering and averaging 3D bounding boxes

    Args:
        bboxes: (N, 9) array of 3D bounding boxes (cx, cy, cz, h, l, w, r, p, y)
        threshold: Threshold for grouping

    Returns:
        averaged_bboxes: (M, 9) array of averaged 3D bounding boxes
    """
    if len(bboxes) == 0:
        return np.empty((0, 9))

    # Extract centroids and apply DBSCAN
    centroids = np.array([bbox[:3] for bbox in bboxes])
    clustering = DBSCAN(eps=threshold, min_samples=1).fit(centroids)

    # Clustering
    labels = clustering.labels_
    unique_labels = set(labels)

    # Averaging
    averaged_bboxes = []
    for label in unique_labels:
        cluster_bboxes = bboxes[labels == label]

        # Skip if there is only one bbox
        if len(cluster_bboxes) < 3:
            continue

        avg_centroid_dim = np.mean(cluster_bboxes[:, :6], axis=0)
        avg_rpy = average_rpy(cluster_bboxes[:, 6:])
        averaged_bboxes.append(np.concatenate([avg_centroid_dim, avg_rpy]))

    return np.array(averaged_bboxes)


def main(args):
    rospy.init_node("CODa_global_3d_bbox_getter", anonymous=True)

    # 3D Bounding Box Publisher for each class
    object_pubs = {
        class_name: rospy.Publisher(
            f"instance_3dbbox/{class_name}", MarkerArray, queue_size=1
        )
        for class_name in classes
    }

    # Data Path
    dataset_path = pathlib.Path(args.dataset)

    # Bbox [cx, cy, cz, l, w, h, r, p, y]
    bboxes = {class_name: np.empty((0, 9)) for class_name in classes}

    # Accumulate 3D Bounding Box
    for sequence in args.sequences:
        pose_file = dataset_path / "poses" / "correct" / f"{sequence}.txt"
        pose_np = np.loadtxt(pose_file, delimiter=" ").reshape(-1, 8)

        timestamp_file = dataset_path / "timestamps" / f"{sequence}.txt"
        timestamp_np = np.loadtxt(timestamp_file, delimiter=" ")

        # Data Path
        bbox_3d_root_dir = dataset_path / "3d_bbox" / "os1" / str(sequence)

        # Main Loop
        print(f"Processing Sequence {sequence}")
        for pose in tqdm(pose_np, total=len(pose_np)):
            frame = np.searchsorted(timestamp_np, pose[0], side="left")

            bbox_3d_file = bbox_3d_root_dir / f"3d_bbox_os1_{sequence}_{frame}.json"

            if not os.path.exists(bbox_3d_file):
                continue

            # Transformation Matrix from Global to LiDAR
            H_lg = np.eye(4)
            H_lg[:3, :3] = R.from_quat(pose[[5, 6, 7, 4]]).as_matrix()
            H_lg[:3, 3] = pose[1:4]

            # Get 3D Bounding Box in Global Frame
            bbox_3d_json = json.load(open(bbox_3d_file, "r"))
            for bbox_3d in bbox_3d_json["3dbbox"]:
                class_name = bbox_3d["classId"]

                if class_name not in classes:
                    continue

                # fmt: off
                bbox = np.array([
                    bbox_3d["cX"], bbox_3d["cY"], bbox_3d["cZ"],
                    bbox_3d["l"], bbox_3d["w"], bbox_3d["h"],
                    bbox_3d["r"], bbox_3d["p"], bbox_3d["y"]
                ])
                # fmt: on
                bbox_global = transform_bbox_3d(bbox, H_lg)
                bboxes[class_name] = np.vstack([bboxes[class_name], bbox_global])

    # Get Averaged Bbox
    averaged_bboxes = {
        class_name: cluster_average_bbox_3d(bboxes[class_name])
        for class_name in classes
    }

    # Publish 3D Bounding Box
    for class_name in classes:
        marker_array = MarkerArray()
        for idx, bbox in enumerate(averaged_bboxes[class_name]):
            bbox_marker = create_bbox_3d_marker(
                bbox_3d=bbox,
                frame_id=global_frame,
                marker_id=idx,
                color=classes[class_name]["color"],
            )
            marker_array.markers.append(bbox_marker)
        clear_marker_array(object_pubs[class_name])
        object_pubs[class_name].publish(marker_array)

    # Save 3D Bounding Box in Global Frame to JSON
    for class_name in classes:
        bbox_3d_json = {"3dbbox": []}
        for idx, bbox in enumerate(averaged_bboxes[class_name]):
            bbox_3d = {
                "classId": class_name,
                "instanceId": class_name + "_" + str(idx),
                "cX": bbox[0],
                "cY": bbox[1],
                "cZ": bbox[2],
                "h": bbox[5],
                "l": bbox[3],
                "w": bbox[4],
                "r": bbox[6],
                "p": bbox[7],
                "y": bbox[8],
            }
            bbox_3d_json["3dbbox"].append(bbox_3d)

        bbox_3d_file = dataset_path / "3d_bbox" / "global" / f"{class_name}.json"
        json.dump(bbox_3d_json, open(bbox_3d_file, "w"), indent=4)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
