"""
Author: Dongmyeong Lee (domlee[at]utexas.edu)
Date:   Apr 22, 2024
Description: Compensate Pointcloud with high-frequency pose
"""

import os
import sys
import pathlib
import argparse
from tqdm import tqdm
from natsort import natsorted
import time
import warnings
import matplotlib.pyplot as plt

import numpy as np

from utils.lie_math import xyz_quat_to_SE3, SE3_to_xyz_quat


def get_closest_poses(poses, ts):
    upper_pose_idx = np.searchsorted(poses[:, 0], ts, side="right")
    lower_pose_idx = upper_pose_idx - 1
    if upper_pose_idx == len(poses) or lower_pose_idx == -1:
        warnings.warn(f"No pose range found for {ts}. Assume there is no motion")
        if upper_pose_idx == len(poses):
            lower_pose_idx = upper_pose_idx = len(poses) - 1
        elif lower_pose_idx == -1:
            lower_pose_idx = upper_pose_idx = 0

    # Get the lower and upper poses
    lower_pose = poses[lower_pose_idx]
    upper_pose = poses[upper_pose_idx]
    return lower_pose, upper_pose


def motion_compensation(pc_file, scan_ts, dt, poses):
    N_HORIZON = 1024
    N_RING = 128

    # TODO: check the N_HORIZON order for time interpolation
    try:
        points = np.fromfile(pc_file, dtype=np.float32).reshape(N_RING, N_HORIZON, -1)
        points = points[:, :, :3]  # x, y, z

    except IOError:
        raise IOError("Could not read the file:", pc_file)

    lower_pose, upper_pose = get_closest_poses(poses, scan_ts)

    # Check if the scan_ts is within the pose range
    if lower_pose[0] == upper_pose[0]:
        comp_pc = points.reshape(-1, 3)
        base_pose = np.array([scan_ts] + list(lower_pose[1:]))
        return comp_pc, base_pose

    # pose at the scan_ts
    lower_SE3 = xyz_quat_to_SE3(lower_pose[1:])
    upper_SE3 = xyz_quat_to_SE3(upper_pose[1:])
    t = (scan_ts - lower_pose[0]) / (upper_pose[0] - lower_pose[0])
    base_SE3 = lower_SE3 + t * (upper_SE3 - lower_SE3)
    base_pose = np.array([scan_ts] + list(SE3_to_xyz_quat(base_SE3)))

    base_matrix = base_SE3.transform()
    inv_base_matrix = np.linalg.inv(base_matrix)

    # time between horizontal scans
    t_values = np.linspace(0, dt, N_HORIZON, endpoint=False)
    comp_pc = np.zeros((N_RING * N_HORIZON, 3), dtype=np.float32)
    comp_pc[:N_RING] = points[:, 0, :]
    for idx, ts in enumerate(t_values[1:], 1):
        timestamp = scan_ts + ts

        # Find the closest pose
        lower_pose, upper_pose = get_closest_poses(poses, timestamp)
        lower_SE3 = xyz_quat_to_SE3(lower_pose[1:])
        upper_SE3 = xyz_quat_to_SE3(upper_pose[1:])

        # Interpolate the pose if the timestamp is in between the lower and upper poses
        curr_SE3 = lower_SE3
        if lower_pose[0] != upper_pose[0]:
            t = (timestamp - lower_pose[0]) / (upper_pose[0] - lower_pose[0])
            curr_SE3 = lower_SE3 + t * (upper_SE3 - lower_SE3)

        relative_matrix = inv_base_matrix @ curr_SE3.transform()
        curr_pc = np.hstack([points[:, idx, :], np.ones((N_RING, 1))])
        comp_pc[idx * N_RING : (idx + 1) * N_RING] = curr_pc @ relative_matrix[:3].T

    # pose at the scan_ts
    return comp_pc, base_pose


def main(args):
    print(f"Compensate Pointcloud for CODa sequence {args.seq}")

    # Pose Data
    dense_poses = np.loadtxt(args.dense_posefile)[:, :8]
    print(f"Total {len(dense_poses)} pose data")

    # Point Cloud Data and its Timestamp
    pc_files = natsorted(list(args.pc_dir.glob("*.bin")))
    timestamps = np.loadtxt(args.timestamp, dtype=np.float64)
    print(f"Total {len(pc_files)} pointcloud files and timestamps")
    assert len(pc_files) == len(timestamps)

    poses = np.zeros((len(timestamps), 8))
    for idx, ts in tqdm(enumerate(timestamps), total=len(timestamps)):
        # motion compensation
        dt = (
            timestamps[idx + 1] - ts
            if idx + 1 < len(timestamps)
            else timestamps[idx] - timestamps[idx - 1]
        )
        comp_pc, pose = motion_compensation(pc_files[idx], ts, dt, dense_poses)
        poses[idx] = pose

        # Save the compensated pointcloud
        pc_outfile = args.out_dir / f"3d_comp_os1_{args.seq}_{idx}.bin"
        comp_pc.astype(np.float32).tofile(pc_outfile)

    # Save the synchronized poses
    np.savetxt(args.posefile, poses, fmt="%.6f %.8f %.8f %.8f %.8f %.8f %.8f %.8f")
    print(f"Saved the synchronized poses to {args.posefile}")


def get_args():
    parser = argparse.ArgumentParser(description="Compensate Pointcloud")
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset directory")
    parser.add_argument("--seq", type=int, help="Sequence number")

    args = parser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset_dir)
    args.pc_dir = args.dataset_dir / "3d_raw" / "os1" / str(args.seq)
    args.timestamp = args.dataset_dir / "timestamps" / f"{args.seq}.txt"
    args.dense_posefile = args.dataset_dir / "poses" / "point_lio" / f"{args.seq}.txt"

    args.out_dir = args.dataset_dir / "3d_comp" / "os1" / str(args.seq)
    args.posefile = args.dataset_dir / "poses" / "point_lio" / f"sync_{args.seq}.txt"
    os.makedirs(args.out_dir, exist_ok=True)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
