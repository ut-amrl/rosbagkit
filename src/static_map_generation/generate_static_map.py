"""
Author:      Dongmyeong Lee (domlee[at]utexas[dot]edu)
Date:        Apr 24, 2024
Description: Generate static map by accumulating pointclouds and downsampling
"""

import pathlib
import argparse
from tqdm import tqdm
from natsort import natsorted
import pathlib

import numpy as np
import open3d as o3d

from src.utils.lie_math import xyz_quat_to_matrix
from src.utils.o3d_visualization import create_o3d_grid


def accumulate_pointcloud(pc_files, pose_np, blind, skip=30):
    """Accumulate pointclouds based on the poses"""
    accumulated_pc_o3d = o3d.geometry.PointCloud()
    for pose, pc_file in tqdm(
        zip(pose_np[::skip], pc_files[::skip]),
        total=len(pose_np // skip),
        desc="Accumulating Pointclouds",
    ):
        Hwl = xyz_quat_to_matrix(pose[1:])
        pc_np = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)[:, :3]
        # Remove points within the blind region
        pc_np = pc_np[np.linalg.norm(pc_np, axis=1) > blind]

        # Transform the pointcloud to the world frame
        pc_world = pc_np @ Hwl[:3, :3].T + Hwl[:3, 3].T

        # Accumulate the pointcloud
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(pc_world)
        accumulated_pc_o3d += pc_o3d

    return accumulated_pc_o3d


def downsample_pointcloud(pc, voxel_size, nb_neighbors, std_ratio):
    """Downsample the pointcloud using voxel grid and remove statistical outliers"""
    print("Downsampling the pointcloud...")
    # Downsample using a voxel grid
    voxel_down_pc = pc.voxel_down_sample(voxel_size=voxel_size)

    # Remove statistical outliers
    clean_pc, _ = voxel_down_pc.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return clean_pc


def main(args):
    print(f"Generating static map for scene: {args.scenes}...")

    all_accumulated_pc_o3d = None
    for pc_dir, pose_file in zip(args.pc_dirs, args.pose_files):
        if not pathlib.Path(pose_file).exists():
            print(f"No Exist: {pose_file}")
            continue

        pose_np = np.loadtxt(pose_file)[:, :8]
        pc_files = natsorted(pathlib.Path(pc_dir).glob("*.bin"))
        if len(pose_np) != len(pc_files):
            print(f"Skipping {pc_dir}, {len(pose_np)} != {len(pc_files)}")
            continue

        accumulated_pc_o3d = accumulate_pointcloud(pc_files, pose_np, args.blind)

        if all_accumulated_pc_o3d is None:
            all_accumulated_pc_o3d = accumulated_pc_o3d
        else:
            all_accumulated_pc_o3d += accumulated_pc_o3d

    # Downsample the accumulated pointcloud
    downsampled_pc_o3d = downsample_pointcloud(
        all_accumulated_pc_o3d, args.voxel_size, args.nb_neighbors, args.std_ratio
    )

    x_min = np.min(np.asarray(accumulated_pc_o3d.points)[:, 0])
    x_max = np.max(np.asarray(accumulated_pc_o3d.points)[:, 0])
    y_min = np.min(np.asarray(accumulated_pc_o3d.points)[:, 1])
    y_max = np.max(np.asarray(accumulated_pc_o3d.points)[:, 1])

    # Save the static map
    if args.extension == "pcd":
        o3d.io.write_point_cloud(args.static_map_file, downsampled_pc_o3d)
    elif args.extension == "npy":
        static_map_np = np.asarray(downsampled_pc_o3d.points, dtype=np.float32)
        np.save(args.static_map_file, static_map_np)
    elif args.extension == "bin":
        static_map_np = np.asarray(downsampled_pc_o3d.points, dtype=np.float32)
        static_map_np.tofile(args.static_map_file)
    else:
        raise ValueError(f"Invalid extension: {args.extension}")
    print(f"{len(downsampled_pc_o3d.points)} points saved to {args.static_map_file}\n")

    if args.visualize:
        print(f"Distance Scale: x={x_max-x_min}, y={y_max-y_min}")
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=10.0, origin=[0, 0, 0]
        )
        grid = create_o3d_grid(x_min, x_max, y_min, y_max, 50)
        o3d.visualization.draw_geometries(
            [downsampled_pc_o3d, coordinate_frame, grid], window_name="Static Map"
        )


def get_args():
    parser = argparse.ArgumentParser(description="Generate static map")
    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="CODa", choices=["CODa", "Wanda"]
    )
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset")
    parser.add_argument("--scenes", type=str, nargs="+", help="Scene names")

    # Static map
    parser.add_argument("--name", type=str, help="Name of the static map")
    parser.add_argument(
        "--extension", type=str, default="pcd", choices=["pcd", "npy", "bin"]
    )

    # Options
    parser.add_argument("--blind", type=float, default=15.0)
    parser.add_argument("--voxel_size", type=float, default=1.0)
    parser.add_argument("--nb_neighbors", type=int, default=100)
    parser.add_argument("--std_ratio", type=float, default=2.0)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset_dir)
    if args.dataset == "CODa":
        args.pc_dirs = [
            args.dataset_dir / "3d_comp" / "os1" / scene for scene in args.scenes
        ]
        args.pose_files = [
            args.dataset_dir / "correct" / f"{scene}.txt" for scene in args.scenes
        ]
        args.static_map_file = (
            args.dataset_dir / "static_map" / f"{args.name}.{args.extension}"
        )
    elif args.dataset == "Wanda":
        args.pc_dirs = [args.dataset_dir / "3d_comp" / scene for scene in args.scenes]
        args.pose_files = [
            args.dataset_dir / "poses" / scene / "os1.txt" for scene in args.scenes
        ]
        args.static_map_file = (
            args.dataset_dir / "static_map" / f"{args.name}.{args.extension}"
        )
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    args.static_map_file.parent.mkdir(parents=True, exist_ok=True)
    args.static_map_file = str(args.static_map_file)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
