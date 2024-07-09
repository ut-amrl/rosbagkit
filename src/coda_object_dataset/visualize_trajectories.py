"""
Author: Dongmyeong Lee (domlee[at]utexas.edu)
Date:   May 25, 2024
Description: Visualize multiple trajectories in Global Frame
"""

import argparse
import pathlib
from natsort import natsorted
import json
import numpy as np
from tqdm import tqdm
import time

import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.utils.o3d_visualization import create_o3d_3d_bbox, create_o3d_grid

# fmt: off
CLASSES = {
    "Tree":               {"id": 0, "color": (0, 1.0, 0)},  # Green
    "Pole":               {"id": 1, "color": (0, 0, 1.0)},  # Blue
    "Bollard":            {"id": 2, "color": (1.0, 0, 0)},  # Red
    "Informational_Sign": {"id": 3, "color": (1.0, 1.0, 0)},  # Yellow
    "Traffic_Sign":       {"id": 4, "color": (0.5, 0, 0.5)},  # Purple
    "Trash_Can":          {"id": 5, "color": (1.0, 0.5, 1.0)},  # Pink
    "Fire_Hydrant":       {"id": 6, "color": (1.0, 0.65, 0)},  # Orange
    "Emergency_Phone":    {"id": 7, "color": (0, 0.5, 0.5)},  # Teal
}
# fmt: on


def plot_trajectories(args):
    fig, ax = plt.subplots()

    # Define the colors for each trajectory
    colors = plt.cm.jet(np.linspace(0, 1, len(args.sequences)))

    for i, seq in enumerate(args.sequences):
        pose_file = args.pose_dir / f"{seq}.txt"
        pose_np = np.loadtxt(pose_file).reshape(-1, 8)

        positions = pose_np[:, 1:4]

        ax.plot(
            positions[:, 0],
            positions[:, 1],
            # color=colors[i],
            color="black",
            linewidth=3,
            # label=f"Seq {seq}",
        )

    # Load and plot the static map
    # pcd = np.asarray(o3d.io.read_point_cloud(args.map_pcd).points)
    # ax.scatter(pcd[:, 0], pcd[:, 1], color="gray", s=1, alpha=0.5, label="Point Cloud")

    train_color = (0, 0, 1)  # Blue for training/validation
    test_color = (1, 0, 0)  # Red for testing

    # Global Annotation
    legend_patches = []
    for class_id in CLASSES:
        global_bbox_file = args.dataset_dir / "3d_bbox" / "global" / f"{class_id}.json"
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

            if bbox[0] <= 0 and class_id in ["Tree", "Pole", "Bollard"]:
                rect = plt.Rectangle(
                    (bbox[0] - bbox[3] / 2, bbox[1] - bbox[4] / 2),
                    bbox[3],
                    bbox[4],
                    linewidth=10,
                    edgecolor=train_color,
                    facecolor="none",
                )
            else:
                rect = plt.Rectangle(
                    (bbox[0] - bbox[3] / 2, bbox[1] - bbox[4] / 2),
                    bbox[3],
                    bbox[4],
                    linewidth=10,
                    edgecolor=test_color,
                    facecolor="none",
                )

            ax.add_patch(rect)

            # rect = plt.Rectangle(
            #     (bbox[0] - bbox[3] / 2, bbox[1] - bbox[4] / 2),
            #     bbox[3],
            #     bbox[4],
            #     linewidth=10,
            #     edgecolor=CLASSES[class_id]["color"],
            #     facecolor="none",
            # )
            # ax.add_patch(rect)

        # class_name = class_id.replace("_", " ")
        # legend_patches.append(
        #     mpatches.Patch(color=CLASSES[class_id]["color"], label=class_name)
        # )

    # Customize plot
    ax.set_xlabel("X (m)", fontsize=25)
    ax.set_ylabel("Y (m)", fontsize=25)
    ax.tick_params(axis="both", which="major", labelsize=25)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    # ax.legend(handles=legend_patches, fontsize=25)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=train_color, lw=4, label="Train/Val"),
            plt.Line2D([0], [0], color=test_color, lw=4, label="Test"),
        ],
        fontsize=25,
    )
    plt.show()


def main_o3d(args):
    # Create an Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.array([1, 1, 1])
    vis.get_render_option().line_width = 50

    # define the colors for each trajectory
    colors = plt.cm.jet(np.linspace(0, 1, len(args.sequences)))

    # Add the trajectories
    for i, seq in enumerate(args.sequences):
        pose_file = args.pose_dir / f"{seq}.txt"
        pose_np = np.loadtxt(pose_file).reshape(-1, 8)

        positions = pose_np[:, 1:4]
        lines = [[j, j + 1] for j in range(len(positions) - 1)]
        colors_list = [colors[i][:3]] * len(lines)

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(positions),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors_list)

        vis.add_geometry(line_set)

    # Load the static map and downsample
    pcd = o3d.io.read_point_cloud(args.map_pcd)
    pcd = pcd.voxel_down_sample(voxel_size=3.0)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    vis.add_geometry(pcd)

    # Add the grid
    x_min = pcd.get_min_bound()[0]
    x_max = pcd.get_max_bound()[0]
    y_min = pcd.get_min_bound()[1]
    y_max = pcd.get_max_bound()[1]
    o3d_grid = create_o3d_grid(x_min, x_max, y_min, y_max, grid_size=10)
    vis.add_geometry(o3d_grid)

    # Global Annotation
    for class_id in CLASSES:
        global_bbox_file = args.dataset_dir / "3d_bbox" / "global" / f"{class_id}.json"
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
            obb = create_o3d_3d_bbox(bbox, color=CLASSES[class_id]["color"])
            vis.add_geometry(obb)

    vis.run()
    vis.destroy_window()


def get_args():
    parser = argparse.ArgumentParser(
        description="Visualize multiple trajectories with different colors"
    )
    parser.add_argument("--dataset_dir", type=str, default="data/CODa")
    parser.add_argument(
        "--sequences",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21],
        help="The sequences to process",
    )
    args = parser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset_dir)
    args.pose_dir = args.dataset_dir / "correct"
    args.map_pcd = str(args.dataset_dir / "static_map" / "all.pcd")

    return args


if __name__ == "__main__":
    args = get_args()
    # main(args)
    plot_trajectories(args)
