"""
Author: Dongmyeong Lee (domlee[at]utexas.edu)
Date:   May 25, 2024
Description: Visualize the accumulated pointcloud map and object instance annotations
"""

import argparse
import pathlib
from natsort import natsorted
import json
import numpy as np
from tqdm import tqdm
import time

import open3d as o3d

from src.utils.o3d_visualization import (
    O3dVisualizer,
    create_o3d_3d_bbox,
    create_o3d_pointcloud,
)
from src.utils.lie_math import xyz_quat_to_matrix
from src.utils.geometry import transform_3d_bbox

# fmt: off
CLASSES = {
    "Tree":               {"id": 0, "color": (0, 1.0, 0)},
    "Pole":               {"id": 1, "color": (0, 0, 1.0)},
    "Bollard":            {"id": 2, "color": (1.0, 0, 0)},
    "Informational_Sign": {"id": 3, "color": (1.0, 1.0, 0)},
    "Traffic_Sign":       {"id": 4, "color": (1.0, 0, 1.0)},
    "Trash_Can":          {"id": 5, "color": (0, 1.0, 1.0)},
    "Fire_Hydrant":       {"id": 6, "color": (1.0, 0.5, 0)},
    "Emergency_Phone":    {"id": 7, "color": (0, 1.0, 0.5)},
}
# fmt: on


def main(args):
    # Visualize
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.get_render_option().point_size = 1.0
    visualizer.get_render_option().background_color = np.array([0, 0, 0])
    visualizer.get_render_option().line_width = 10

    # Load the accumulated pointcloud map
    map_pcd = o3d.io.read_point_cloud(args.map_pcd)
    map_pcd.paint_uniform_color([1.0, 1.0, 1.0])
    visualizer.add_geometry(map_pcd)

    # Global Annotation
    for class_name in CLASSES:
        global_bbox_file = (
            args.dataset_dir / "3d_bbox" / "global" / f"{class_name}.json"
        )
        if not global_bbox_file.exists():
            print(f"No exist: {global_bbox_file}")
            continue

        global_bboxes = json.load(open(global_bbox_file))
        for instance in global_bboxes["instances"]:
            bbox = np.array(
                [
                    instance[attr]
                    for attr in ["cX", "cY", "cZ", "l", "w", "h", "r", "p", "y"]
                ]
            )
            obb = create_o3d_3d_bbox(bbox, CLASSES[class_name]["color"])
            visualizer.add_geometry(obb)

    # # CODa Annotation
    # for sequence in args.sequences:
    #     pose_file = args.pose_dir / f"{sequence}.txt"
    #     if not pose_file.exists():
    #         print(f"No exist: {pose_file}")
    #         continue
    #     pose_np = np.loadtxt(pose_file).reshape(-1, 8)

    #     for frame in tqdm(range(len(pose_np)), desc=f"Sequence {sequence}"):
    #         os1_3d_bbox_file = (
    #             args.annotation_dir
    #             / str(sequence)
    #             / f"3d_bbox_os1_{sequence}_{frame}.json"
    #         )
    #         if not os1_3d_bbox_file.exists():
    #             continue

    #         # Load pose
    #         Twl = xyz_quat_to_matrix(pose_np[frame][1:])

    #         # Load 3D bounding boxes
    #         bboxes = {class_name: [] for class_name in CLASSES}
    #         for instance in json.load(open(os1_3d_bbox_file))["3dbbox"]:
    #             class_name = instance["classId"].replace(" ", "_")
    #             if class_name in CLASSES:
    #                 bbox = np.array(
    #                     [
    #                         instance[attr]
    #                         for attr in ["cX", "cY", "cZ", "l", "w", "h", "r", "p", "y"]
    #                     ]
    #                 )
    #                 global_bbox = transform_3d_bbox(bbox, Twl)
    #                 bboxes[class_name].append(global_bbox)

    #                 obb = create_o3d_3d_bbox(
    #                     global_bbox, color=CLASSES[class_name]["color"]
    #                 )
    #                 visualizer.add_geometry(obb)

    visualizer.run()
    visualizer.destroy_window()


def get_args():
    parser = argparse.ArgumentParser(
        description="Visualize the accumulated pointcloud map and object instance annotations"
    )
    parser.add_argument("--dataset_dir", type=str, default="data/CODa")
    parser.add_argument(
        "--sequences",
        type=int,
        nargs="+",
        # default=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21],
        default=[2],
        help="The sequences to process",
    )
    args = parser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset_dir)
    args.pose_dir = args.dataset_dir / "correct"
    # args.pose_dir = args.dataset_dir / "poses" / "ct-icp_sync"
    args.annotation_dir = args.dataset_dir / "3d_bbox" / "os1"
    args.map_pcd = str(args.dataset_dir / "static_map" / "2510.pcd")

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
