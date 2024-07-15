"""
Author: Dongmyeong Lee (domlee[at]utexas.edu)
Date:   Apr 22, 2024
Description: Generate depth images from stereo images and pointcloud
"""

import os
import argparse
import pathlib
from natsort import natsorted

import numpy as np
import cv2

from src.utils.camera import load_extrinsics, load_cam_params
from src.utils.pose_interpolator import PoseInterpolator
from src.utils.transforms import xyz_quat_to_matrix

from utils.depth import project_volume_to_depth
from utils.misc import alternating_indices


def process_frame(idx, data, args, pose_interpolator):
    img_ts = data["timestamps"][idx]

    # accumulated compensated pointcloud in the world frame
    accumulated_pc = []
    close_pc_idx = np.argmin(np.abs(data["poses"][:, 0] - img_ts))
    for i in alternating_indices(args.window_size):
        pc_idx = close_pc_idx + i
        if pc_idx < 0 or pc_idx >= len(data["pc_files"]):
            continue

        pc = np.fromfile(data["pc_files"][pc_idx], dtype=np.float32).reshape(-1, 4)
        pc = pc[:, :3]
        Hwl = xyz_quat_to_matrix(data["poses"][pc_idx, 1:])

        pc_world = pc @ Hwl[:3, :3].T + Hwl[:3, 3].T  # Transform (Lidar -> World)
        accumulated_pc.append(pc_world)

    # Get the transformation from the world to the rectified image plane
    Hwl_img = pose_interpolator.get_interpolated_transform(img_ts)
    rect_left, rect_right = np.eye(4), np.eye(4)
    rect_left[:3, :3] = data["cam0_params"]["R"]
    rect_right[:3, :3] = data["cam1_params"]["R"]
    Hrw_left = rect_left @ data["cam0_extrinsic"] @ np.linalg.inv(Hwl_img)
    Hrw_right = rect_right @ data["cam1_extrinsic"] @ np.linalg.inv(Hwl_img)

    # compute the depth image by projecting the pointcloud to the image plane
    img_left = cv2.imread(data["img_left_files"][idx])
    img_right = cv2.imread(data["img_right_files"][idx])

    depth_left = project_volume_to_depth(
        img_left,
        accumulated_pc,
        Hrw_left,
        data["cam0_params"]["P"],
        volume=args.volume,
        visualize=True,
    )
    depth_right = project_volume_to_depth(
        img_right,
        accumulated_pc,
        Hrw_right,
        data["cam1_params"]["P"],
        volume=args.volume,
        visualize=True,
    )


def main(args):
    data = load_data(args)
    pose_interpolator = PoseInterpolator(data["poses"])

    for idx in range(len(data["img_left_files"])):
        if idx % 100 == 0:
            process_frame(idx, data, args, pose_interpolator)


def load_data(args):
    # Load the point cloud files and poses
    pc_files = list(map(str, natsorted(args.pc_dir.glob("*.bin"))))
    poses = np.loadtxt(args.pose_file)  # timestamp, x, y, z, qw, qx, qy, qz
    assert len(pc_files) == len(poses), f"{len(pc_files)} != {len(poses)}"
    print(f"Loaded {len(pc_files)} pointclouds and corresponding poses")

    # Load the images and timestamps
    img_left_files = list(map(str, natsorted(list(args.img_left_dir.glob("*.png")))))
    img_right_files = list(map(str, natsorted(list(args.img_right_dir.glob("*.png")))))
    timestamps = np.loadtxt(args.timestamps)
    print(f"Loaded {len(img_right_files)} left / right images")
    assert (
        len(img_left_files) == len(img_right_files) == len(timestamps)
    ), f"{len(img_left_files)} != {len(img_right_files)} != {len(timestamps)}"

    # Load the calibrations
    cam0_extrinsic = load_extrinsics(args.cam0_extrinsic)
    cam1_extrinsic = load_extrinsics(args.cam1_extrinsic)
    cam0_params = load_cam_params(args.cam0_intrinsics)
    cam1_params = load_cam_params(args.cam1_intrinsics)

    data = {
        "pc_files": pc_files,
        "poses": poses,
        "img_left_files": img_left_files,
        "img_right_files": img_right_files,
        "timestamps": timestamps,
        "cam0_extrinsic": cam0_extrinsic,
        "cam1_extrinsic": cam1_extrinsic,
        "cam0_params": cam0_params,
        "cam1_params": cam1_params,
    }
    return data


def get_args():
    parser = argparse.ArgumentParser(description="Generate depth images")
    parser.add_argument(
        "--dataset_dir", type=str, default="data/CODa", help="Path to the dataset"
    )
    parser.add_argument("--seq", type=str, default="1", help="Sequence number")
    parser.add_argument(
        "--window_size", type=int, default=1, help="window size for pc accumulation"
    )
    parser.add_argument(
        "--volume", type=float, default=0.05, help="point volume for depth estimation"
    )
    args = parser.parse_args()

    # input
    args.dataset_dir = pathlib.Path(args.dataset_dir)

    args.pc_dir = args.dataset_dir / "3d_comp" / "os1" / args.seq
    # args.pose_file = args.dataset_dir / "poses" / "os1" / f"{args.seq}.txt"
    args.pose_file = args.dataset_dir / "correct" / f"{args.seq}.txt"

    args.img_left_dir = args.dataset_dir / "2d_rect" / "cam0" / args.seq
    args.img_right_dir = args.dataset_dir / "2d_rect" / "cam1" / args.seq
    args.timestamps = args.dataset_dir / "timestamps" / f"{args.seq}.txt"

    args.calib_dir = args.dataset_dir / "calibrations" / args.seq
    args.cam0_extrinsic = args.calib_dir / "calib_os1_to_cam0.yaml"
    args.cam1_extrinsic = args.calib_dir / "calib_os1_to_cam1.yaml"
    args.cam0_intrinsics = args.calib_dir / "calib_cam0_intrinsics.yaml"
    args.cam1_intrinsics = args.calib_dir / "calib_cam1_intrinsics.yaml"

    # output
    args.left_outdir = args.dataset_dir / "2d_depth" / "cam0" / args.seq
    args.right_outdir = args.dataset_dir / "2d_depth" / "cam1" / args.seq
    args.left_outdir.mkdir(parents=True, exist_ok=True)
    args.right_outdir.mkdir(parents=True, exist_ok=True)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
