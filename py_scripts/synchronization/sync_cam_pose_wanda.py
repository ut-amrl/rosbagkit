import os
import sys
import pathlib
import argparse
from natsort import natsorted
import warnings

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from utils.coda_utils import load_extrinsic_matrix, load_camera_params
from utils.lie_math import (
    SE3_to_xyz_quat,
    matrix_to_SE3,
    interpolate_SE3,
    matrix_to_xyz_quat,
    xyz_quat_to_matrix,
)


def linear_interpolation(timestamps, poses):
    interpolated_poses = []
    for ts in timestamps:
        upper_idx = np.searchsorted(poses[:, 0], ts, side="right")

        if upper_idx == len(poses):
            interpolated_poses.append(poses[-1, 1:])
        elif upper_idx == 0:
            interpolated_poses.append(poses[0, 1:])
        else:
            upper_pose = poses[upper_idx]
            lower_pose = poses[upper_idx - 1]

            # Interpolate the poses
            interpolated_pose = interpolate_SE3(lower_pose, upper_pose, ts)
            interpolated_poses.append(interpolated_pose)

    return interpolated_poses


def load_data(args):
    print(f"Loading data from {args.dataset_dir} for {args.scene}")

    # Load the LiDAR poses
    poses = np.loadtxt(args.pose_file)  # timestamp, x, y, z, qw, qx, qy, qz

    # Load the images and its timestamps
    img_left_files = natsorted(list(args.img_left_dir.glob("*.jpg")))
    img_left_timestamps = np.loadtxt(args.img_left_timestamp)
    assert len(img_left_files) == len(img_left_timestamps)
    print(f"Loaded {len(img_left_files)} left images")

    img_right_files = natsorted(list(args.img_right_dir.glob("*.jpg")))
    img_right_timestamps = np.loadtxt(args.img_right_timestamp)
    assert len(img_right_files) == len(img_right_timestamps)
    print(f"Loaded {len(img_right_files)} right images")

    # Load the camera intrinsics
    H_lc_left = load_extrinsic_matrix(args.cam_left_extrinsic)
    H_lc_right = load_extrinsic_matrix(args.cam_right_extrinsic)

    data = {
        "poses": poses,
        "img_left_timestamps": img_left_timestamps,
        "img_right_timestamps": img_right_timestamps,
        "H_lc_left": H_lc_left,
        "H_lc_right": H_lc_right,
    }
    return data


def main(args):
    data = load_data(args)

    # Interpolate the LiDAR poses to the image timestamps
    poses = data["poses"]
    left_timestamps = data["img_left_timestamps"]
    right_timestamps = data["img_right_timestamps"]
    os_left_ts_poses = linear_interpolation(left_timestamps, poses)
    os_right_ts_poses = linear_interpolation(right_timestamps, poses)

    # Transform the LiDAR poses to the camera poses using the extrinsics (Left)
    H_cl_left = np.linalg.inv(data["H_lc_left"])
    cam_left_poses = [
        matrix_to_xyz_quat(xyz_quat_to_matrix(H_lw) @ H_cl_left)
        for H_lw in os_left_ts_poses
    ]
    cam_left_poses = np.hstack((left_timestamps[:, None], cam_left_poses))
    np.savetxt(
        args.cam_left_pose_file,
        cam_left_poses,
        fmt="%.6f %.8f %.8f %.8f %.8f %.8f %.8f %.8f",
    )

    # Transform the LiDAR poses to the camera poses using the extrinsics (Right)
    H_cl_right = np.linalg.inv(data["H_lc_right"])
    cam_right_poses = [
        matrix_to_xyz_quat(xyz_quat_to_matrix(H_lw) @ H_cl_right)
        for H_lw in os_right_ts_poses
    ]
    cam_right_poses = np.hstack((right_timestamps[:, None], cam_right_poses))
    np.savetxt(
        args.cam_right_pose_file,
        cam_right_poses,
        fmt="%.6f %.8f %.8f %.8f %.8f %.8f %.8f %.8f",
    )


def get_args():
    parser = argparse.ArgumentParser(description="Synchronize camera pose for Wanda")
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset directory")
    parser.add_argument("--scene", type=str, help="Scene name")
    args = parser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset_dir)
    args.pose_dir = args.dataset_dir / "poses" / args.scene
    args.pose_file = args.pose_dir / "os1.txt"

    args.img_left_dir = args.dataset_dir / "2d_rect" / args.scene / "left"
    args.img_right_dir = args.dataset_dir / "2d_rect" / args.scene / "right"
    timestamp_dir = args.dataset_dir / "timestamps" / args.scene
    args.img_left_timestamp = timestamp_dir / "img_left.txt"
    args.img_right_timestamp = timestamp_dir / "img_right.txt"

    args.calib_dir = args.dataset_dir / "calibrations" / args.scene
    args.cam_left_extrinsic = args.calib_dir / "os_to_cam_left.yaml"
    args.cam_right_extrinsic = args.calib_dir / "os_to_cam_right.yaml"
    args.cam_left_intrinsics = args.calib_dir / "cam_left_intrinsics.yaml"
    args.cam_right_intrinsics = args.calib_dir / "cam_right_intrinsics.yaml"

    args.cam_left_pose_file = args.pose_dir / "cam_left.txt"
    args.cam_right_pose_file = args.pose_dir / "cam_right.txt"
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
