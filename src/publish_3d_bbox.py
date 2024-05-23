"""
Author:      Dongmyeong Lee (domlee[at]utexas[dot]edu)
Date:        Feb 20, 2024
Description: Debug 3D bbox annotation
"""

import os
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2
import tf2_ros

from utils.ros_viz_utils import create_3d_bbox_marker, clear_marker_array
from utils.ros_utils import wait_for_subscribers
from utils.math_utils import average_rpy
from utils.msg_converter import np_to_pointcloud2

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

PC_TOPIC = "/os_points"


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
        default=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22],
        # default=[0],
        help="Sequence ID",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0,
        help="Rate for publishing Pointcloud",
    )

    args = parser.parse_args()

    args.dataset_dir = Path(args.dataset)
    # args.pose_dir = args.dataset_dir / "poses" / "keyframe"
    args.pose_dir = args.dataset_dir / "poses"
    args.timestamp_dir = args.dataset_dir / "timestamps"
    args.bbox_3d_dir = args.dataset_dir / "3d_bbox" / "os1"
    args.global_bbox_3d_dir = args.dataset_dir / "3d_bbox" / "global"
    args.pc_dir = args.dataset_dir / "3d_comp" / "os1"
    return args


def load_3d_annotaions(bbox_3d_file: dict) -> np.ndarray:
    """
    Load 3D Bounding Box in Global Frame

    Args:
        bbox_3d_file: Path to 3D Bounding Box JSON file

    Returns:
        bboxes: (N, 9) array of 3D Bounding Box in Global Frame
    """
    os.path.exists(bbox_3d_file)

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
        bboxes[class_name] = np.vstack([bboxes[class_name], bbox])

    return bboxes


def main():
    args = get_args()

    # Visualize 3D Bounding Box annotation by frame
    rospy.init_node("CODa_3d_bbox_getter", anonymous=True)

    # 3D Bounding Box Publisher for each class
    instance_pubs = {
        class_name: rospy.Publisher(
            f"instance_3dbbox/{class_name}", MarkerArray, queue_size=1
        )
        for class_name in CLASSES
    }
    pc_pub = rospy.Publisher(PC_TOPIC, PointCloud2, queue_size=1)
    wait_for_subscribers(list(instance_pubs.values()))

    # Publish 3D Bounding Box annotation
    global_bboxes = {class_name: np.empty((0, 9)) for class_name in CLASSES}
    for sequence in args.sequences:
        pose_file = args.pose_dir / f"{sequence}.txt"
        pose_np = np.loadtxt(pose_file).reshape(-1, 8)

        timestamp_file = args.timestamp_dir / f"{sequence}.txt"
        timestamp_np = np.loadtxt(timestamp_file)

        # Data Path
        bbox_3d_root_dir = args.bbox_3d_dir / f"{sequence}"

        # Main Loop
        print(f"Processing Sequence {sequence}, Total Frame: {len(pose_np)}")
        for pose in tqdm(pose_np, total=len(pose_np)):
            frame = np.searchsorted(timestamp_np, pose[0], side="left")

            # Clear all markers
            for class_name in instance_pubs.keys():
                clear_marker_array(instance_pubs[class_name])

            bbox_3d_file = bbox_3d_root_dir / f"3d_bbox_os1_{sequence}_{frame}.json"
            if not os.path.exists(bbox_3d_file):
                continue

            bboxes = load_3d_annotaions(bbox_3d_file)

            # Publish 3D Bounding Box for each class
            for class_name in instance_pubs.keys():
                marker_array = MarkerArray()
                for idx, bbox in enumerate(bboxes[class_name]):
                    bbox_marker = create_3d_bbox_marker(
                        bbox_3d=bbox,
                        frame_id="base_link",
                        marker_id=idx,
                        color=CLASSES[class_name]["color"],
                    )
                    marker_array.markers.append(bbox_marker)
                instance_pubs[class_name].publish(marker_array)

            # Publish Pointcloud
            pc_file = (
                args.pc_dir / str(sequence) / f"3d_comp_os1_{sequence}_{frame}.bin"
            )
            if os.path.exists(pc_file):
                pc_np = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 3)
                pc_msg = np_to_pointcloud2(pc_np, "x y z", "base_link")
                pc_pub.publish(pc_msg)
                if args.rate > 0:
                    time.sleep(1 / args.rate)

    print("Finished Accumulating 3D Bounding Box")


if __name__ == "__main__":
    main()
