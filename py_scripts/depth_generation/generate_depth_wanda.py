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

import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from utils.coda_utils import load_extrinsic_matrix, load_camera_params
from utils.image import get_disparity_map, draw_epipolar_lines
from utils.depth import fill_depth_bins, densify_depth_image, save_depth_image
from utils.camera import project_to_image, project_to_rectified
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
    left_pose: np.ndarray,
    right_pose: np.ndarray,
    pc_world_window: deque[np.ndarray],
    data: dict,
    debug: bool = False,
):
    assert img_left.shape == img_right.shape
    H, W = img_left.shape[:2]

    H_cw_left = xyz_quat_to_matrix(left_pose[1:])
    H_cw_right = xyz_quat_to_matrix(right_pose[1:])
    H_wc_left = np.linalg.inv(H_cw_left)
    H_wc_right = np.linalg.inv(H_cw_right)

    # 1. Depth map from accumulated point clouds
    depth_bins_left = np.full((H, W, 3), np.nan, dtype=np.float32)
    depth_bins_right = np.full((H, W, 3), np.nan, dtype=np.float32)
    for pc_world in pc_world_window:
        # Project the point cloud to the left and right images
        pc_img_left, pc_depth_left, valid_left = project_to_rectified(
            img_left,
            pc_world,
            H_wc_left,
            data["cam_left"]["R"],
            data["cam_left"]["P"],
        )
        pc_img_right, pc_depth_right, valid_right = project_to_rectified(
            img_right,
            pc_world,
            H_wc_right,
            data["cam_right"]["R"],
            data["cam_right"]["P"],
        )

        # Accumulate the depth bins
        depth_bins_left = fill_depth_bins(
            depth_bins_left, pc_img_left, pc_depth_left, option="min"
        )
        depth_bins_right = fill_depth_bins(
            depth_bins_right, pc_img_right, pc_depth_right, option="min"
        )

    if debug:
        non_nan_left = np.sum(~np.isnan(depth_bins_left[:, :, 2]))
        non_nan_right = np.sum(~np.isnan(depth_bins_right[:, :, 2]))
        print("ratio of non-nan pixels (left): ", non_nan_left / (H * W))
        print("ratio of non-nan pixels (right): ", non_nan_right / (H * W))

    # 2. Compute the depth map with disparity map
    get_disparity_map(img_left, img_right)

    densified_depth_left = densify_depth_image(depth_bins_left) 
    densified_depth_right = densify_depth_image(depth_bins_right)

    return densified_depth_left, densified_depth_right


def load_data(args):
    # Load the point cloud files and poses
    pc_files = list(map(str, natsorted(args.pc_dir.glob("*.bin"))))
    poses = np.loadtxt(args.pose_file)  # timestamp, x, y, z, qw, qx, qy, qz
    assert len(pc_files) == len(poses)
    print(f"Loaded {len(pc_files)} pointclouds and poses")
    print(f"pose file: {args.pose_file}")

    # Load the images and timestamps
    img_left_files = list(map(str, natsorted(args.img_left_dir.glob("*.jpg"))))
    left_poses = np.loadtxt(args.left_pose_file)
    assert len(img_left_files) == len(left_poses)
    print(f"Loaded {len(img_left_files)} left images and poses")

    img_right_files = list(map(str, natsorted(args.img_right_dir.glob("*.jpg"))))
    right_poses = np.loadtxt(args.right_pose_file)
    assert len(img_right_files) == len(right_poses)
    print(f"Loaded {len(img_right_files)} right images and poses")
    assert len(img_left_files) == len(right_poses)

    # Load the calibrations
    cam_left = load_camera_params(args.cam_left_intrinsics)
    cam_right = load_camera_params(args.cam_right_intrinsics)

    data = {
        "pc_files": pc_files,
        "poses": poses,
        "img_left_files": img_left_files,
        "img_right_files": img_right_files,
        "left_poses": left_poses,
        "right_poses": right_poses,
        "cam_left": cam_left,
        "cam_right": cam_right,
    }
    return data


def main(args):
    print(f"Generating depth images for {args.scene}")

    data = load_data(args)

    # Initialize the point cloud window
    pc_world_window = deque(maxlen=args.window)
    last_pc_idx = -1

    for i in range(len(data["img_left_files"]))[-10:]:
        img_left = cv2.imread(str(data["img_left_files"][i]))
        img_right = cv2.imread(str(data["img_right_files"][i]))
        left_pose = data["left_poses"][i]
        right_pose = data["right_poses"][i]

        if (left_pose[0] - right_pose[0]) > 0.1:
            warnings.warn(f"Not synchronized poses: {left_pose[0]} vs {right_pose[0]}")

        # Accumulate the point clouds
        upper_pc_idx = np.searchsorted(data["poses"][:, 0], left_pose[0], side="right")
        for pc_idx in range(last_pc_idx + 1, upper_pc_idx):
            pc = np.fromfile(data["pc_files"][pc_idx], dtype=np.float32).reshape(-1, 3)
            H_lw = xyz_quat_to_matrix(data["poses"][pc_idx][1:])

            # Transform the point cloud to the world frame
            pc_world = pc @ H_lw[:3, :3].T + H_lw[:3, 3].T
            pc_world_window.append(pc_world)
        last_pc_idx = upper_pc_idx - 1

        # Compute the stereo depth with the accumulated point clouds
        depth_left, depth_right = compute_stereo_depth(
            img_left,
            img_right,
            left_pose,
            right_pose,
            pc_world_window,
            data,
            args.debug,
        )

        # save the depth images
        left_file = str(args.left_outdir / f"2d_depth_left_{i}.png")
        right_file = str(args.right_outdir / f"2d_depth_right_{i}.png")
        save_depth_image(depth_left, left_file)
        save_depth_image(depth_right, right_file)
        print(f"Saved depth images: {left_file}, {right_file}")

        if args.debug:
            left_rgbd_file = str(args.left_rgbd_outdir / f"2d_rgbd_left_{i}.png")
            right_rgbd_file = str(args.right_rgbd_outdir / f"2d_rgbd_right_{i}.png")
            visualize_rgbd_image(img_left, depth_left, 0.3, outfile=left_rgbd_file)
            visualize_rgbd_image(img_right, depth_right, 0.3, outfile=right_rgbd_file)
            print(f"Saved rgbd images: {left_rgbd_file}, {right_rgbd_file}")


def get_args():
    parser = argparse.ArgumentParser(description="Generate depth images")
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset")
    parser.add_argument("--scene", type=str, help="Scene name")
    parser.add_argument("--window", type=int, default=10, help="window size")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset_dir)

    args.pc_dir = args.dataset_dir / "3d_comp" / args.scene
    args.pose_file = args.dataset_dir / "poses" / args.scene / "os1.txt"

    args.img_left_dir = args.dataset_dir / "2d_rect" / args.scene / "left"
    args.img_right_dir = args.dataset_dir / "2d_rect" / args.scene / "right"
    args.left_pose_file = args.dataset_dir / "poses" / args.scene / "cam_left.txt"
    args.right_pose_file = args.dataset_dir / "poses" / args.scene / "cam_right.txt"

    args.calib_dir = args.dataset_dir / "calibrations" / args.scene
    args.cam_left_intrinsics = args.calib_dir / "cam_left_intrinsics.yaml"
    args.cam_right_intrinsics = args.calib_dir / "cam_right_intrinsics.yaml"

    args.left_outdir = args.dataset_dir / "2d_depth" / args.scene / "left"
    args.right_outdir = args.dataset_dir / "2d_depth" / args.scene / "right"
    args.left_outdir.mkdir(parents=True, exist_ok=True)
    args.right_outdir.mkdir(parents=True, exist_ok=True)

    args.left_rgbd_outdir = args.dataset_dir / "2d_rgbd" / args.scene / "left"
    args.right_rgbd_outdir = args.dataset_dir / "2d_rgbd" / args.scene / "right"
    args.left_rgbd_outdir.mkdir(parents=True, exist_ok=True)
    args.right_rgbd_outdir.mkdir(parents=True, exist_ok=True)
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
