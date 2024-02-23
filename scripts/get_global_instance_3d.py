"""
Author:      Dongmyeong Lee (domlee[at]utexas[dot]edu)
Date:        Feb 20, 2024
Description: Get global 3D bounding box from annotation of CODa dataset
"""
import os
from pathlib import Path
import argparse
import json
import jsbeautifier
from tqdm import tqdm

import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN

import rospy
from visualization_msgs.msg import Marker, MarkerArray

from utils.geometry import transform_bbox_3d
from utils.ros_viz_utils import create_bbox_3d_marker, clear_marker_array
from utils.ros_utils import wait_for_subscribers
from utils.math_utils import average_rpy

# fmt: off
CLASSES = {
    "Tree":               {"id": 0, "color": (0, 1.0, 0)},
    "Pole":               {"id": 1, "color": (0, 0, 1.0)},
    "Bollard":            {"id": 2, "color": (1.0, 0, 0)},
    "Informational_Sign": {"id": 3, "color": (1.0, 1.0, 0)},
    "Traffic_Sign":       {"id": 4, "color": (1.0, 0, 1.0)},
    "Trash_Can":          {"id": 5, "color": (0, 1.0, 1.0)},
    "Fire_Hydrant":       {"id": 6, "color": (0.5, 0.5, 0.5)},
    "Emergency_Phone":    {"id": 7, "color": (0.5, 0.5, 0.5)},
}
# fmt: on


def get_args():
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
        # default=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21],
        default=[0],
        help="Sequence ID",
    )
    parser.add_argument(
        "--ros",
        action="store_true",
        help="Publish 3D Bounding Box as ROS Marker",
    )

    args = parser.parse_args()

    args.dataset_dir = Path(args.dataset)
    args.pose_dir = args.dataset_dir / "poses"
    args.timestamp_dir = args.dataset_dir / "timestamps"
    args.bbox_3d_dir = args.dataset_dir / "3d_bbox" / "os1"
    args.global_bbox_3d_dir = args.dataset_dir / "3d_bbox" / "global"
    return args


def cluster_average_bbox_3d(bboxes: np.ndarray, threshold: float = 1.5) -> np.ndarray:
    """
    Clustering and averaging 3D bounding boxes

    Args:
        bboxes: (N, 9) array of 3D bounding boxes (cX, cY, cZ, l, w, h, r, p, y)
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

        # # Skip if there is only one bbox
        # if len(cluster_bboxes) < 3:
        # continue

        avg_centroid_dim = np.mean(cluster_bboxes[:, :6], axis=0)
        avg_rpy = average_rpy(cluster_bboxes[:, 6:])
        averaged_bboxes.append(np.concatenate([avg_centroid_dim, avg_rpy]))

    return np.array(averaged_bboxes)


def load_3d_annotaions(pose: np.ndarray, bbox_3d_file: dict) -> np.ndarray:
    """
    Load 3D Bounding Box in Global Frame

    Args:
        pose: (8,) array of pose (timestamp, x, y, z, qw, qx, qy, qz)
        bbox_3d_file: Path to 3D Bounding Box JSON file

    Returns:
        bboxes: (N, 9) array of 3D Bounding Box in Global Frame
    """
    if not os.path.exists(bbox_3d_file):
        return {class_name: np.empty((0, 9)) for class_name in CLASSES}

    # Transformation Matrix from Global to LiDAR
    T_lg = np.eye(4)
    T_lg[:3, 3] = pose[1:4]
    T_lg[:3, :3] = R.from_quat(pose[[5, 6, 7, 4]]).as_matrix()

    bboxes = {class_name: np.empty((0, 9)) for class_name in CLASSES}

    bbox_3d_json = json.load(open(bbox_3d_file))
    for instance in bbox_3d_json["3dbbox"]:
        class_name = instance["classId"].replace(" ", "_")
        if class_name not in CLASSES:
            continue

        bbox = np.array(
            [
                instance[attr]
                for attr in ["cX", "cY", "cZ", "l", "w", "h", "r", "p", "y"]
            ]
        )
        bbox_global = transform_bbox_3d(bbox, T_lg)
        bboxes[class_name] = np.vstack([bboxes[class_name], bbox_global])

    return bboxes


def main():
    args = get_args()

    # Accumulate 3D Bounding Box
    global_bboxes = {class_name: np.empty((0, 9)) for class_name in CLASSES}
    for sequence in args.sequences:
        pose_file = args.pose_dir / f"{sequence}.txt"
        pose_np = np.loadtxt(pose_file, delimiter=" ").reshape(-1, 8)

        timestamp_file = args.timestamp_dir / f"{sequence}.txt"
        timestamp_np = np.loadtxt(timestamp_file, delimiter=" ")

        # Data Path
        bbox_3d_root_dir = args.bbox_3d_dir / f"{sequence}"

        # Main Loop
        print(f"Processing Sequence {sequence}")
        for pose in tqdm(pose_np, total=len(pose_np)):
            frame = np.searchsorted(timestamp_np, pose[0], side="left")

            bbox_3d_file = bbox_3d_root_dir / f"3d_bbox_os1_{sequence}_{frame}.json"
            bboxes = load_3d_annotaions(pose, bbox_3d_file)

            global_bboxes = {
                class_name: np.vstack([global_bboxes[class_name], bboxes[class_name]])
                for class_name in global_bboxes.keys()
            }
    print("Finished Accumulating 3D Bounding Box")

    # Get Averaged Bbox
    averaged_bboxes = {
        class_name: cluster_average_bbox_3d(class_bboxes)
        for class_name, class_bboxes in global_bboxes.items()
    }

    # Save 3D Bounding Box in Global Frame to JSON
    opts = jsbeautifier.default_options()
    opts.indent_size = 2
    for class_name in averaged_bboxes.keys():
        class_json = {"class": class_name, "instances": []}
        for idx, bbox in enumerate(averaged_bboxes[class_name]):
            instance = {
                "id": idx,
                "cX": 
                "cY": bbox[1],
                "cZ": bbox[2],
                "l": bbox[3],
                "w": bbox[4],
                "h": bbox[5],
                "r": bbox[6],
                "p": bbox[7],
                "y": bbox[8],
            }
            class_json["instances"].append(instance)

        class_file = args.global_bbox_3d_dir / f"{class_name}.json"
        os.makedirs(os.path.dirname(class_file), exist_ok=True)
        with open(class_file, "w") as f:
            f.write(jsbeautifier.beautify(json.dumps(class_json), opts))

    # ROS Visualization
    if args.ros:
        rospy.init_node("CODa_global_3d_bbox_getter", anonymous=True)

        # 3D Bounding Box Publisher for each class
        object_pubs = {
            class_name: rospy.Publisher(
                f"instance_3dbbox/{class_name}", MarkerArray, queue_size=1
            )
            for class_name in CLASSES
        }
        wait_for_subscribers(list(object_pubs.values()))

        for class_name in object_pubs.keys():
            marker_array = MarkerArray()
            for idx, bbox in enumerate(averaged_bboxes[class_name]):
                bbox_marker = create_bbox_3d_marker(
                    bbox_3d=bbox,
                    frame_id="map",
                    marker_id=idx,
                    color=CLASSES[class_name]["color"],
                )
                marker_array.markers.append(bbox_marker)
            clear_marker_array(object_pubs[class_name])
            object_pubs[class_name].publish(marker_array)


if __name__ == "__main__":
    main()
