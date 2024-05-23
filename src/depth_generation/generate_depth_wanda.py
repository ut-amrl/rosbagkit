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

from utils.coda_utils import load_extrinsic_matrix, load_camera_params
from utils.transforms import xyz_quat_to_matrix


from src.depth_generation.depth_renderer_o3d import DepthRenderer


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

    print("Camera left intrinsics:")
    print(cam_left)

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

    width, height = data["cam_left"]["img_size"]
    print(f"Image size: {width} x {height}")

    # Initialize the depth renderer
    depth_renderer = DepthRenderer(
        width=width,
        height=height,
        voxel_size=args.voxel_size,
        maxlen=args.maxlen,
        visible=args.debug,
    )
    projection_matrix = data["cam_left"]["P"]
    print(f"Projection matrix:\n{projection_matrix[:3,:3]}")

    depth_renderer.update_view_point_intrinsics(projection_matrix[:3, :3])
    # depth_renderer.update_view_point_intrinsics(data["cam_left"]["K"])
    rect_mat = np.eye(4)
    rect_mat[:3, :3] = data["cam_left"]["R"]
    print(f"Rectification matrix:\n{rect_mat}")

    last_pc_idx = -1
    for i in range(len(data["img_left_files"])):
        print(i)
        left_pose = data["left_poses"][i]
        timestamp = left_pose[0]
        Hwc = xyz_quat_to_matrix(left_pose[1:])
        Hcw = np.linalg.inv(Hwc)

        # Accumulate the point clouds
        upper_pc_idx = np.searchsorted(data["poses"][:, 0], timestamp, side="right")
        for pc_idx in range(last_pc_idx + 1, upper_pc_idx):
            pc = np.fromfile(data["pc_files"][pc_idx], dtype=np.float32).reshape(-1, 3)
            Hwl = xyz_quat_to_matrix(data["poses"][pc_idx][1:])

            # Transform the point cloud to the world frame
            pc_world = pc @ Hwl[:3, :3].T + Hwl[:3, 3].T
            depth_renderer.add_pointcloud(pc_world)
        last_pc_idx = upper_pc_idx - 1

        depth_renderer.update_view_point_extrinsics(rect_mat @ Hcw)

        if args.debug:
            depth_renderer.run()

        filename = str(args.left_outdir / f"2d_depth_left_{i}.png")
        depth_renderer.capture_depth_image(filename)

    # # save the depth images
    # left_file = str(args.left_outdir / f"2d_depth_left_{i}.png")
    # right_file = str(args.right_outdir / f"2d_depth_right_{i}.png")
    # save_depth_image(depth_left, left_file)
    # save_depth_image(depth_right, right_file)
    # print(f"Saved depth images: {left_file}, {right_file}")

    # if args.debug:
    #     left_rgbd_file = str(args.left_rgbd_outdir / f"2d_rgbd_left_{i}.png")
    #     right_rgbd_file = str(args.right_rgbd_outdir / f"2d_rgbd_right_{i}.png")
    #     visualize_rgbd_image(img_left, depth_left, 0.3, outfile=left_rgbd_file)
    #     visualize_rgbd_image(img_right, depth_right, 0.3, outfile=right_rgbd_file)
    #     print(f"Saved rgbd images: {left_rgbd_file}, {right_rgbd_file}")


def get_args():
    parser = argparse.ArgumentParser(description="Generate depth images")
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset")
    parser.add_argument("--scene", type=str, help="Scene name")
    parser.add_argument(
        "--voxel_size", type=float, default=0.1, help="voxel_size of the point"
    )
    parser.add_argument(
        "--maxlen", type=int, default=20, help="window size for accumulation"
    )
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
