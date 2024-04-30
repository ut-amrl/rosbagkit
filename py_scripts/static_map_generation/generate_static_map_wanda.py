"""
Author:      Dongmyeong Lee (domlee[at]utexas[dot]edu)
Date:        Apr 24, 2024
Description: Generate static map by accumulating pointclouds and downsampling
"""

import os
import sys
import pathlib
import argparse
from tqdm import tqdm
import time
from natsort import natsorted
import pathlib

import numpy as np
import open3d as o3d


sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from utils.lie_math import xyz_quat_to_matrix


def accumulate_pointcloud(pc_files, pose_np, blind):
    """Accumulate pointclouds based on the poses"""
    print("Accumulating pointclouds...")
    accumulated_pc_o3d = o3d.geometry.PointCloud()
    for pose, pc_file in tqdm(zip(pose_np, pc_files), total=len(pose_np), leave=False):
        H_lw = xyz_quat_to_matrix(pose[1:])
        pc_np = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 3)
        # Remove points within the blind region
        pc_np = pc_np[np.linalg.norm(pc_np, axis=1) > blind]

        # Transform the pointcloud to the world frame
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(pc_np)
        pc_o3d.transform(H_lw)
        accumulated_pc_o3d += pc_o3d

    return accumulated_pc_o3d


def downsample_pointcloud(pc, voxel_size, nb_neighbors, std_ratio):
    """Downsample the pointcloud using voxel grid and remove statistical outliers"""
    print("Downsampling the pointcloud...")
    # Downsample using a voxel grid
    voxel_down_pc = pc.voxel_down_sample(voxel_size=voxel_size)

    # Remove statistical outliers
    clean_pc, ind = voxel_down_pc.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return clean_pc


def main(args):
    print(f"Generating static map for scene: {args.scene}...")

    pose_np = np.loadtxt(args.pose_file)[:, :8]
    pc_files = natsorted(pathlib.Path(args.pc_dir).glob("*.bin"))
    assert len(pose_np) == len(pc_files), f"{len(pose_np)} != {len(pc_files)}"

    accumulated_pc = accumulate_pointcloud(pc_files, pose_np, args.blind)
    clean_pc = downsample_pointcloud(
        accumulated_pc, args.voxel_size, args.nb_neighbors, args.std_ratio
    )

    # Save the static map
    static_map_np = np.asarray(clean_pc.points, dtype=np.float32)
    static_map_np.tofile(args.static_map_file)
    print(f"{len(static_map_np)} points saved to {args.static_map_file}")

    o3d.visualization.draw_geometries([clean_pc], window_name="Static Map")


def get_args():
    parser = argparse.ArgumentParser(description="Generate static map for wanda")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/dongmyeong/Projects/datasets/SARA",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="gq_appld_south_tour_01_2024-03-14-10-08-34",
        help="Scene name",
    )
    parser.add_argument("--blind", type=float, default=30.0)
    parser.add_argument("--voxel_size", type=float, default=1.0)
    parser.add_argument("--nb_neighbors", type=int, default=1000)
    parser.add_argument("--std_ratio", type=float, default=1.0)
    args = parser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset_dir)
    args.pose_file = args.dataset_dir / "poses" / args.scene / "os1.txt"
    args.pc_dir = args.dataset_dir / "3d_comp" / args.scene
    args.static_map_file = args.dataset_dir / "static_map" / f"{args.scene}.bin"
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
