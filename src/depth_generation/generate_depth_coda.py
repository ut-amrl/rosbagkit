"""
Author: Dongmyeong Lee (domlee[at]utexas.edu)
Date:   Apr 22, 2024
Description: Generate depth images from stereo images and pointcloud
"""

import os
import sys
import argparse
import pathlib
from natsort import natsorted
from collections import deque
import warnings
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import matplotlib.pyplot as plt

from utils.coda_utils import load_extrinsic_matrix, load_camera_params
from utils.depth import fill_depth_bins, densify_depth_image, save_depth_image
from utils.projection import project_to_image
from utils.transforms import xyz_quat_to_matrix
from utils.visualization import (
    visualize_pointcloud,
    visualize_rgbd_image,
    visualize_normalized_image,
    draw_points_on_image,
)


def compute_stereo_depth(
    img_left: np.ndarray,
    img_right: np.ndarray,
    pc_world_window: deque[np.ndarray],
    pose: np.ndarray,  # timestamp, x, y, z, qw, qx, qy, qz
    data: dict,
):
    assert img_left.shape == img_right.shape
    H, W = img_left.shape[:2]

    # 1. Project the point cloud to the camera
    H_lw = xyz_quat_to_matrix(pose[1:])
    H_wl = np.linalg.inv(H_lw)

    depth_bins_left = np.full((H, W, 3), np.nan, dtype=np.float32)
    depth_bins_right = np.full((H, W, 3), np.nan, dtype=np.float32)

    for pc_world in pc_world_window:
        pc_img_left, pc_depth_left, valid_left = project_to_image(
            img_left,
            pc_world,
            data["cam_left"]["H_lc"] @ H_wl,  # H_wc = H_lc @ H_wl
            data["cam_left"]["K"],
            data["cam_left"]["D"],
        )
        pc_img_right, pc_depth_right, valid_right = project_to_image(
            img_right,
            pc_world,
            data["cam_right"]["H_lc"] @ H_wl,  # H_wc = H_lc @ H_wl
            data["cam_right"]["K"],
            data["cam_right"]["D"],
        )

        # Accumulate the depth bins
        depth_bins_left = fill_depth_bins(depth_bins_left, pc_img_left, pc_depth_left)
        depth_bins_right = fill_depth_bins(
            depth_bins_right, pc_img_right, pc_depth_right
        )

    # 2. Compute the depth map with disparity map

    # 3. Densify the depth map with Inverse Depth Fusion
    densified_depth_left = densify_depth_image(depth_bins_left)
    densified_depth_right = densify_depth_image(depth_bins_right)

    return densified_depth_left, densified_depth_right


def load_data(args):
    # Load the point cloud files and poses
    pc_files = natsorted(list(args.pc_dir.glob("*.bin")))
    poses = np.loadtxt(args.pose_file)  # timestamp, x, y, z, qw, qx, qy, qz
    assert len(pc_files) == len(poses)
    print(f"Loaded {len(pc_files)} pointclouds and corresponding poses")

    # Load the images and timestamps
    img_left_files = natsorted(list(args.img_left_dir.glob("*.jpg")))
    img_right_files = natsorted(list(args.img_right_dir.glob("*.jpg")))
    timestamps = np.loadtxt(args.timestamps)
    print(f"Loaded {len(img_right_files)} left / right images")
    assert len(img_left_files) == len(img_right_files) == len(timestamps)

    # Load the calibrations
    cam_left = load_camera_params(args.cam_left_intrinsics)
    cam_right = load_camera_params(args.cam_right_intrinsics)
    cam_left.update({"H_lc": load_extrinsic_matrix(args.cam_left_extrinsic)})
    cam_right.update({"H_lc": load_extrinsic_matrix(args.cam_right_extrinsic)})

    data = {
        "pc_files": pc_files,
        "poses": poses,
        "img_left_files": img_left_files,
        "img_right_files": img_right_files,
        "timestamps": timestamps,
        "cam_left": cam_left,
        "cam_right": cam_right,
    }
    return data


def main(args):
    data = load_data(args)

    # Initialize the point cloud window
    pc_world_window = deque(maxlen=args.window)
    last_pc_idx = -1

    for idx, ts in enumerate(data["timestamps"]):
        img_left = cv2.imread(str(data["img_left_files"][idx]))
        img_right = cv2.imread(str(data["img_right_files"][idx]))

        # Accumulate the point clouds
        upper_pc_idx = np.searchsorted(data["poses"][:, 0], ts, side="right")
        for pc_idx in range(last_pc_idx + 1, upper_pc_idx):
            print(f"Update the point cloud window with {pc_idx}")
            pc = np.fromfile(data["pc_files"][pc_idx], dtype=np.float32).reshape(-1, 3)
            pose = data["poses"][pc_idx]
            H_lw = xyz_quat_to_matrix(pose[1:])

            # Transform the point cloud to the world frame
            pc_lidar = np.hstack((pc, np.ones((pc.shape[0], 1))))
            pc_world = pc_lidar @ H_lw[:3].T  # Nx4 @ 4x3 -> Nx3
            pc_world_window.append(pc_world)
        last_pc_idx = upper_pc_idx - 1

        if len(pc_world_window) < args.window:
            continue

        # Compute the stereo depth with the accumulated point clouds
        depth_left, depth_right = compute_stereo_depth(
            img_left,
            img_right,
            pc_world_window,
            data["poses"][last_pc_idx],
            data,
        )

        # save the depth images
        frame = data["img_left_files"][idx].stem.split("_")[-1]
        left_file = str(args.left_outdir / f"2d_depth_cam0_{args.seq}_{frame}.png")
        right_file = str(args.right_outdir / f"2d_depth_cam1_{args.seq}_{frame}.png")
        save_depth_image(depth_left, left_file)
        save_depth_image(depth_right, right_file)


def get_args():
    parser = argparse.ArgumentParser(description="Generate depth images")
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset")
    parser.add_argument("--seq", type=str, help="seq name")
    parser.add_argument("--window", type=int, default=10, help="window size")
    args = parser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset_dir)

    args.pc_dir = args.dataset_dir / "3d_comp" / "os1" / args.seq
    args.pose_file = args.dataset_dir / "poses" / "os1" / f"{args.seq}.txt"

    args.img_left_dir = args.dataset_dir / "2d_raw" / "cam0" / args.seq
    args.img_right_dir = args.dataset_dir / "2d_raw" / "cam1" / args.seq
    args.timestamps = args.dataset_dir / "timestamps" / f"{args.seq}.txt"

    args.calib_dir = args.dataset_dir / "calibrations" / args.seq
    args.cam_left_extrinsic = args.calib_dir / "calib_os1_to_cam0.yaml"
    args.cam_left_intrinsics = args.calib_dir / "calib_cam0_intrinsics.yaml"
    args.cam_right_extrinsic = args.calib_dir / "calib_os1_to_cam1.yaml"
    args.cam_right_intrinsics = args.calib_dir / "calib_cam1_intrinsics.yaml"

    args.left_outdir = args.dataset_dir / "2d_depth" / "cam0" / args.seq
    args.right_outdir = args.dataset_dir / "2d_depth" / "cam1" / args.seq
    args.left_outdir.mkdir(parents=True, exist_ok=True)
    args.right_outdir.mkdir(parents=True, exist_ok=True)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
